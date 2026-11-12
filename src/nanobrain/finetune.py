import argparse
import datetime
import json
import logging
import os
import signal
import threading
import time
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn as nn
import wandb
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode

import nanobrain.misc as misc
from nanobrain.data import Fomo300kDataset, InfiniteSampler, synth_lesion
from nanobrain.model import ViTMAE3D
from nanobrain.train import ignore_sigterm, make_lr_schedule, sample_transform

logger = logging.getLogger()

ROOT = Path(__file__).parents[2]
# brainmarks-smri data folder, shared with the eval
DATA_ROOT = Path(os.environ["BRAINMARKS_SMRI_DATA"])
DEFAULT_CONFIG = Path(__file__).parent / "config/default_finetune.yaml"

# synth_lesion presets and how often each is drawn for a lesion image
LESION_PRESETS = {
    "ellipsoid": (0.15, dict(roughness=0.0, texture=0.0)),
    "lobulated": (0.35, dict(roughness=0.5, texture=0.3)),
    "ring": (0.25, dict(roughness=0.5, texture=0.3, ring_prob=1.0)),
    "multifocal": (0.25, dict(num_lesions=(3, 8), radius=(2.0, 8.0), roughness=0.3, texture=0.2)),
}


def main(args: DictConfig):
    # setup
    process_start_time = time.monotonic()
    assert int(os.environ.get("WORLD_SIZE", 1)) == 1, "distributed training not supported"
    device = torch.device(args.device)
    misc.random_seed(args.seed)

    if args.name and not args.output_dir.endswith(args.name):
        args.output_dir = f"{args.output_dir}/{args.name}"
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_cfg_path = output_dir / "config.yaml"
    if out_cfg_path.exists():
        prev_cfg = OmegaConf.load(out_cfg_path)
        assert args == prev_cfg, "current config doesn't match previous config"
    else:
        OmegaConf.save(args, out_cfg_path)

    misc.setup_logging(logger, output_dir / "log.txt")

    logger.info("finetuning nanobrain")
    logger.info(f"start: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"cwd: {Path.cwd()}")
    logger.info(f"git: {misc.git_info()}")
    if device.type == "cuda":
        logger.info(f"gpu: {torch.cuda.get_device_name(device)}")
    logger.info(f"torch: {torch.__version__} (cuda {torch.version.cuda})")
    logger.info(f"config:\n{OmegaConf.to_yaml(args)}")

    summary_path = output_dir / "summary.json"
    if summary_path.exists() and json.loads(summary_path.read_text())["done"]:
        logger.info(f"training already done, see {summary_path}")
        return

    if args.wandb:
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.name,
            notes=args.notes,
            config=OmegaConf.to_container(args),
        )

    misc.snapshot_source(ROOT / "src", output_dir / "source.tar.gz")

    # model
    model = ViTMAE3D.from_config(args)
    assert model.finetune_heads, "set model_kwargs.finetune_heads"
    if args.init_ckpt:
        # pretraining checkpoints have no finetuning heads
        ckpt = torch.load(args.init_ckpt, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        assert not unexpected, f"unexpected keys {unexpected}"
        assert all(key.startswith(("age_head.", "lesion_head.")) for key in missing), missing
        logger.info(f"loaded pretrained weights from {args.init_ckpt}")
    model.to(device)

    # train the last finetune_blocks blocks, the final norm and the heads. never the decoder
    trainable = ["norm.", "age_head.", "lesion_head."]
    num_blocks = len(model.blocks)
    finetune_blocks = num_blocks if args.finetune_blocks is None else args.finetune_blocks
    trainable += [f"blocks.{ii}." for ii in range(num_blocks - finetune_blocks, num_blocks)]
    if finetune_blocks == num_blocks:
        trainable += ["patch_embed.", "pos_embed.", "cls_token"]
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith(tuple(trainable))
    logger.info(f"model:\n{model}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"num trainable params: {num_params / 1e6:.1f}M ({finetune_blocks} blocks)")

    batch_flops = count_batch_flops(args, model, device)
    logger.info(f"flops per batch (fwd + bwd, approx): {batch_flops / 1e12:.2f} TFLOP")

    if args.compile:
        # forward_finetune isn't model.forward, so compile the blocks in place
        for block in model.blocks:
            block.compile()

    # optimizer
    total_batch_size = args.batch_size * args.accum_iter
    logger.info(
        f"total batch size: {total_batch_size} = {args.batch_size} images x {args.accum_iter} accum"
    )

    if not args.get("lr"):
        args.lr = args.base_lr * total_batch_size / 256
        logger.info(f"lr: {args.lr:.2e} = {args.base_lr:.2e} x {total_batch_size} / 256")
    else:
        logger.info(f"lr: {args.lr:.2e}")

    param_groups = misc.get_param_groups(model)
    misc.update_lr(param_groups, args.lr)
    misc.update_wd(param_groups, args.weight_decay)
    # cast or else it corrupts the checkpoint
    betas = tuple(args.betas) if args.betas is not None else None
    optimizer = torch.optim.AdamW(param_groups, betas=betas, fused=args.fused_adamw)

    lr_schedule = make_lr_schedule(
        args.lr, args.min_lr, args.total_steps, args.warmup_steps, args.decay_steps
    )
    logger.info(
        f"lr schedule: total steps = {args.total_steps}, warmup = {args.warmup_steps}, "
        f"decay = {args.decay_steps}"
    )

    # loss scaling not needed for bfloat16 (according to timm)
    if args.amp and args.amp_dtype != "bfloat16":
        loss_scaler = torch.GradScaler(device.type)
    else:
        loss_scaler = None

    # load checkpoint/resume training
    start_step, prev_wall_time = misc.load_model(args, model, optimizer, loss_scaler)

    # data loader. one endless pass over shuffled epochs, resumed at the next unseen image
    dataset = Fomo300kDataset(DATA_ROOT / "fomo300k", args.train_filelist, with_age=True)
    excluded = [f"derivatives/{name}/" for name in args.age_exclude_collections]
    for ii, path in enumerate(dataset.paths):
        if path.startswith(tuple(excluded)):
            dataset.ages[ii] = np.nan
    # age targets are standardized with the train split stats
    args.age_mean = float(np.nanmean(dataset.ages))
    args.age_std = float(np.nanstd(dataset.ages))
    logger.info(
        f"train dataset: {len(dataset)} images, age known for "
        f"{np.isfinite(dataset.ages).mean():.0%}, mean {args.age_mean:.1f} std {args.age_std:.1f}"
    )
    sampler = InfiniteSampler(
        len(dataset), seed=args.seed, start=start_step * args.accum_iter * args.batch_size
    )
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=list,  # no collate, variable shapes
        pin_memory=True,
        drop_last=True,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        worker_init_fn=ignore_sigterm,
    )
    batch_transform = make_batch_transform(args)

    # on SIGTERM (e.g. preemption) finish the current step, save and exit. resumes when relaunched
    stop_event = threading.Event()
    signal.signal(signal.SIGTERM, lambda signum, frame: stop_event.set())

    stats = train_finetune(
        args,
        model,
        train_loader,
        batch_transform,
        optimizer,
        loss_scaler,
        lr_schedule,
        start_step,
        prev_wall_time,
        process_start_time,
        device,
        batch_flops,
        stop_event,
    )

    step = stats["step"]
    images = step * args.accum_iter * args.batch_size
    summary = {
        "step": step,
        "images": images,
        "epochs": images / len(dataset),
        "train_flops": step * args.accum_iter * batch_flops,
        "wall_seconds": stats["wall_time"],
        "start_step": start_step,
        "done": stats["done"],
        "stopped_by_signal": stop_event.is_set(),
        "final_loss": stats["loss"],
        "final_age_loss": stats["age_loss"],
        "final_lesion_loss": stats["lesion_loss"],
        "num_params": num_params,
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "cpus": len(os.sched_getaffinity(0)),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "git": misc.git_info(),
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"summary:\n{json.dumps(summary, indent=2)}")
    wall_time_str = datetime.timedelta(seconds=int(stats["wall_time"]))
    if stats["done"]:
        logger.info(f"done! step {step}, wall time: {wall_time_str}")
    else:
        logger.info(f"stopped by signal at step {step}. wall time: {wall_time_str}")


def train_finetune(
    args: DictConfig,
    model: nn.Module,
    data_loader: Iterable,
    batch_transform: Callable,
    optimizer: torch.optim.Optimizer,
    loss_scaler: torch.GradScaler | None,
    lr_schedule: np.ndarray,
    start_step: int,
    prev_wall_time: float,
    start_time: float,
    device: torch.device,
    batch_flops: float,
    stop_event: threading.Event,
) -> dict:
    model.train()

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("grad", misc.SmoothedValue())
    metric_logger.add_meter("loss", misc.SmoothedValue())
    metric_logger.add_meter("age_loss", misc.SmoothedValue())
    metric_logger.add_meter("lesion_loss", misc.SmoothedValue())
    metric_logger.add_meter(
        "vol/s", misc.SmoothedValue(window_size=1, fmt="{value:.0f} ({global_avg:.0f})")
    )
    metric_logger.add_meter(
        "tflop/s", misc.SmoothedValue(window_size=1, fmt="{value:.0f} ({global_avg:.0f})")
    )
    metric_logger.add_meter(
        "watts", misc.SmoothedValue(window_size=1, fmt="{value:.0f} ({global_avg:.0f})")
    )

    throughput = misc.ThroughputMeter(
        {"vol": args.batch_size, "tflop": batch_flops / 1e12}, device=device
    )

    # steps count optimizer updates, batches count loader batches
    total_steps = args.total_steps if not args.debug else start_step + 10
    print_freq = args.print_freq if not args.debug else 1
    print_freq_batches = print_freq * args.accum_iter
    num_batches = total_steps * args.accum_iter
    start_batch = start_step * args.accum_iter

    amp_dtype = getattr(torch, args.amp_dtype)
    use_cuda = device.type == "cuda"
    output_dir = Path(args.output_dir)
    log_path = output_dir / "log.json"
    # time limit on wall time from the start of main, summed over restarts
    time_limit = args.time_limit * 60 if args.time_limit else float("inf")
    step = start_step
    done = False

    optimizer.zero_grad()

    batches = metric_logger.log_every(
        data_loader,
        print_freq_batches,
        "Finetune:",
        total_steps=num_batches,
        start_step=start_batch,
    )
    for batch_idx, batch in enumerate(batches, start=start_batch):
        need_update = (batch_idx + 1) % args.accum_iter == 0
        log_step = (batch_idx + 1) % print_freq_batches == 0
        lr = float(lr_schedule[step])
        if need_update:
            misc.update_lr(optimizer.param_groups, lr)

        if use_cuda:
            batch = misc.send_data(batch, device)

        batch = batch_transform(batch)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=args.amp):
            age_loss, lesion_loss = model.forward_finetune(
                batch["image"],
                batch["mask"],
                batch["lesion"],
                batch["age"],
                num_tokens=args.num_tokens,
            )
        loss = args.age_weight * age_loss + args.lesion_weight * lesion_loss

        loss_for_log = loss.detach()
        torch._assert_async(torch.isfinite(loss_for_log), "non-finite loss")

        grad_norm = misc.backward_step(
            loss / args.accum_iter,
            optimizer,
            scaler=loss_scaler,
            need_update=need_update,
            max_norm=args.clip_grad,
        )

        # log tensors without syncing, values are only read on log steps
        metric_logger.update(
            loss=loss_for_log, age_loss=age_loss.detach(), lesion_loss=lesion_loss.detach()
        )
        if need_update:
            metric_logger.update(lr=lr, grad=grad_norm)

        throughput.step()
        if log_step:
            metric_logger.update(**throughput.compute())
            log_stats = {
                "step": step + 1,
                "loss": metric_logger.loss.value,
                "age_loss": metric_logger.age_loss.value,
                "lesion_loss": metric_logger.lesion_loss.value,
                "lr": lr,
                "grad": metric_logger.grad.value,
                "vol_per_sec": metric_logger.meters["vol/s"].value,
                "tflop_per_sec": metric_logger.meters["tflop/s"].value,
                "watts": metric_logger.meters["watts"].value,
            }
            with log_path.open("a") as f:
                print(json.dumps(log_stats), file=f)
            if args.wandb:
                wandb.log({f"train/{k}": v for k, v in log_stats.items()}, step=step + 1)

        if not need_update:
            continue
        step += 1
        wall_time = prev_wall_time + time.monotonic() - start_time
        done = step == total_steps or wall_time >= time_limit
        is_last = done or stop_event.is_set()

        is_save_step = step % args.checkpoint_period == 0 or step in (args.checkpoint_steps or [])
        if is_save_step or is_last:
            misc.save_model(args, step, model, optimizer, loss_scaler, wall_time)

        if is_last:
            break

    logger.info(f"Averaged stats: {metric_logger}")
    return {
        "step": step,
        "done": done,
        "wall_time": prev_wall_time + time.monotonic() - start_time,
        "loss": metric_logger.loss.median,
        "age_loss": metric_logger.age_loss.median,
        "lesion_loss": metric_logger.lesion_loss.median,
    }


def make_batch_transform(args: DictConfig):
    preset_names = list(LESION_PRESETS)
    preset_probs = torch.tensor([LESION_PRESETS[name][0] for name in preset_names])

    def fn(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        images = []
        masks = []
        lesions = []
        for sample in batch:
            sample = sample_transform(
                sample,
                target_spacing=args.target_spacing,
                grid_size=tuple(args.grid_size),
                clip_quantiles=tuple(args.clip_quantiles),
            )
            image = sample["image"]
            mask = sample["mask"]
            lesion = torch.zeros_like(image)
            if torch.rand(()) < args.lesion_prob:
                name = preset_names[int(torch.multinomial(preset_probs, 1))]
                image, lesion = synth_lesion(image, mask, **LESION_PRESETS[name][1])
            images.append(image)
            masks.append(mask)
            lesions.append(lesion)

        ages = torch.tensor([sample["age"] for sample in batch], dtype=torch.float32)
        ages = (ages - args.age_mean) / args.age_std
        return {
            "image": torch.stack(images),
            "mask": torch.stack(masks),
            "lesion": torch.stack(lesions),
            "age": ages.to(images[0].device, non_blocking=True),
        }

    return fn


def count_batch_flops(args: DictConfig, model: nn.Module, device: torch.device) -> float:
    # fwd + bwd flops of one batch, on dummy images where every patch is in the mask. with all
    # tokens (num_tokens null), assumes ~3500 in-brain tokens per image, the p8 average-ish
    amp_dtype = getattr(torch, args.amp_dtype)
    num_tokens = args.num_tokens or 3584
    images = torch.zeros(args.batch_size, *args.grid_size, device=device)
    masks = torch.ones(args.batch_size, *args.grid_size, device=device)
    lesions = torch.zeros(args.batch_size, *args.grid_size, device=device)
    ages = torch.zeros(args.batch_size, device=device)
    flop_counter = FlopCounterMode(display=False)
    with flop_counter:
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=args.amp):
            age_loss, lesion_loss = model.forward_finetune(
                images, masks, lesions, ages, num_tokens=num_tokens
            )
        (age_loss + lesion_loss).backward()
    model.zero_grad(set_to_none=True)
    return flop_counter.get_total_flops()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--overrides", type=str, default=None, nargs="+")
    args = parser.parse_args()
    cfg = OmegaConf.load(DEFAULT_CONFIG)
    if args.config:
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.load(args.config))
    if args.overrides:
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.from_dotlist(args.overrides))
    main(cfg)
