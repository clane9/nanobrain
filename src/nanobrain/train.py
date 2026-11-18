import argparse
import copy
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
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode

import nanobrain.misc as misc
from nanobrain.data import (
    Fomo300kDataset,
    InfiniteSampler,
    brle_to_dense,
    make_augment,
    make_preprocess,
)
from nanobrain.masking import make_view_sampler
from nanobrain.model import build_model, patchify3d, select_tokens
from nanobrain.visualization import plot_feature_pca

logger = logging.getLogger()

ROOT = Path(__file__).parents[2]
# brainmarks-smri data folder, shared with the eval
DATA_ROOT = Path(os.environ["BRAINMARKS_SMRI_DATA"])
DEFAULT_CONFIG = Path(__file__).parent / "config/default_pretrain.yaml"


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
    (output_dir / "plots").mkdir(exist_ok=True)
    out_cfg_path = output_dir / "config.yaml"
    if out_cfg_path.exists():
        prev_cfg = OmegaConf.load(out_cfg_path)
        assert args == prev_cfg, "current config doesn't match previous config"
    else:
        OmegaConf.save(args, out_cfg_path)

    misc.setup_logging(logger, output_dir / "log.txt")

    logger.info("pretraining nanobrain")
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
    model = build_model(args)
    model.to(device)
    logger.info(f"model:\n{model}")
    # trained params only (student encoder, predictor, head), the teacher is an EMA copy
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"num params: {num_params / 1e6:.1f}M")

    # data pipeline: shared preprocessing, then independent augmentation per view
    patch_grid_size = tuple(size // args.patch_size for size in args.grid_size)
    view_sampler = make_view_sampler(args.views, patch_grid_size)
    preprocess = make_preprocess(
        tuple(args.clip_quantiles), args.target_spacing, tuple(args.grid_size)
    )
    student_augment = make_augment(args.student_augment)
    teacher_augment = make_augment(args.teacher_augment)
    logger.info(f"view sampler: {view_sampler}")
    logger.info(f"preprocess: {preprocess}")
    logger.info(f"student augment: {student_augment}")
    logger.info(f"teacher augment: {teacher_augment}")

    batch_flops = count_batch_flops(args, model, view_sampler, device)
    logger.info(f"flops per batch (teacher fwd, student fwd + bwd): {batch_flops / 1e12:.2f} TFLOP")

    # compile the two steps separately (as capi), the module keeps its state dict keys
    forward_teacher = model.forward_teacher
    forward_student = model.forward_student
    if args.compile:
        forward_teacher = torch.compile(forward_teacher)
        forward_student = torch.compile(forward_student)

    # optimizer
    # batch size counts sequences, following mae_st which scales by repeated samples
    total_batch_size = args.batch_size * args.num_samples * args.accum_iter
    logger.info(
        f"total batch size: {total_batch_size} = "
        f"{args.batch_size} images x {args.num_samples} samples x {args.accum_iter} accum"
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
    # linear from ema_momentum to ema_momentum_end (I-JEPA 0.996 -> 1), constant if null
    momentum_end = args.ema_momentum_end if args.ema_momentum_end is not None else args.ema_momentum
    momentum_schedule = np.linspace(args.ema_momentum, momentum_end, args.total_steps)
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
    dataset = Fomo300kDataset(DATA_ROOT / "fomo300k", args.train_filelist)
    logger.info(f"train dataset: {len(dataset)} images")
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
    batch_transform = make_batch_transform(
        args, model.encoder.coord_grid, view_sampler, preprocess, student_augment, teacher_augment
    )

    # on SIGTERM (e.g. preemption) finish the current step, save and exit. resumes when relaunched
    stop_event = threading.Event()
    signal.signal(signal.SIGTERM, lambda signum, frame: stop_event.set())

    stats = train(
        args,
        model,
        forward_teacher,
        forward_student,
        train_loader,
        batch_transform,
        optimizer,
        loss_scaler,
        lr_schedule,
        momentum_schedule,
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


def train(
    args: DictConfig,
    model: nn.Module,
    forward_teacher: Callable,
    forward_student: Callable,
    data_loader: Iterable,
    batch_transform: Callable,
    optimizer: torch.optim.Optimizer,
    loss_scaler: torch.GradScaler | None,
    lr_schedule: np.ndarray,
    momentum_schedule: np.ndarray,
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
    last_wall_time = prev_wall_time

    optimizer.zero_grad()

    batches = metric_logger.log_every(
        data_loader,
        print_freq_batches,
        "Train:",
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

        # run volume processing on gpu, not cpu workers, ~25x faster
        batch = batch_transform(batch)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=args.amp):
            targets, teacher_state = forward_teacher(
                batch["teacher_patches"], batch["teacher_coord"]
            )
            loss, student_state = forward_student(
                batch["student_patches"],
                batch["student_coord"],
                batch["teacher_coord"],
                targets,
                batch["token_weights"],
            )

        loss_for_log = loss.detach()
        torch._assert_async(torch.isfinite(loss_for_log), "non-finite loss")

        grad_norm = misc.backward_step(
            loss / args.accum_iter,
            optimizer,
            scaler=loss_scaler,
            need_update=need_update,
            max_norm=args.clip_grad,
        )

        if need_update:
            model.update_teacher(float(momentum_schedule[step]))

        # log tensors without syncing, values are only read on log steps
        monitors = {**teacher_state, **student_state}
        metric_logger.update(loss=loss_for_log, **monitors)
        if need_update:
            metric_logger.update(lr=lr, grad=grad_norm)

        throughput.step()
        if log_step:
            metric_logger.update(**throughput.compute())
            log_stats = {
                "step": step + 1,
                "loss": metric_logger.loss.value,
                "lr": lr,
                "grad": metric_logger.grad.value,
                **{name: metric_logger.meters[name].value for name in monitors},
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

        # pca of teacher features on a few images from the current batch
        # not on stop by signal, just save and exit
        is_plot_step = args.plot_period and (step % args.plot_period == 0 or done)
        if args.plot_num_images and is_plot_step:
            plot_path = plot_features(args, model, batch["image"], batch["mask"], step, amp_dtype)
            if args.wandb:
                wandb.log({"train/feature_pca": wandb.Image(str(plot_path))}, step=step)

        is_save_step = step % args.checkpoint_period == 0 or step in (args.checkpoint_steps or [])
        # crossed one of the checkpoint_minutes since the last step
        is_minutes_step = any(
            last_wall_time < minutes * 60 <= wall_time for minutes in args.checkpoint_minutes or []
        )
        last_wall_time = wall_time
        if is_save_step or is_minutes_step or is_last:
            misc.save_model(args, step, model, optimizer, loss_scaler, wall_time)

        if is_last:
            break

    logger.info(f"Averaged stats: {metric_logger}")
    return {
        "step": step,
        "done": done,
        "wall_time": prev_wall_time + time.monotonic() - start_time,
        "loss": metric_logger.loss.median,
    }


def ignore_sigterm(worker_id: int):
    # `timeout` signals the whole process group. workers keep loading until the main process exits
    signal.signal(signal.SIGTERM, signal.SIG_IGN)


def make_lr_schedule(
    lr: float, min_lr: float, total_steps: int, warmup_steps: int, decay_steps: int
) -> np.ndarray:
    warmup = np.linspace(0.0, lr, warmup_steps, endpoint=False)
    stable = np.full(total_steps - warmup_steps - decay_steps, lr)
    decay = np.linspace(lr, min_lr, decay_steps)
    return np.concatenate([warmup, stable, decay])


def make_batch_transform(
    args: DictConfig,
    coord_grid: Tensor,
    view_sampler: Callable,
    preprocess: nn.Module,
    student_augment: nn.Module,
    teacher_augment: nn.Module,
):
    """Raw fomo300k samples to student and teacher tokens, on gpu."""
    patch_size = args.patch_size

    def fn(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        images, student_images, teacher_images, masks = [], [], [], []
        for sample in batch:
            sample = {
                "values": sample["values"].float() * sample["scale"],
                "mask": brle_to_dense(sample["mask_rle"], shape=sample["shape"]),
                "affine": sample["affine"],
            }
            sample = preprocess(sample)
            images.append(sample["image"])
            masks.append(sample["mask"])
            student_images.append(student_augment(sample)["image"])
            teacher_images.append(teacher_augment(sample)["image"])
        images = torch.stack(images)
        masks = torch.stack(masks)
        student_images = torch.stack(student_images)
        teacher_images = torch.stack(teacher_images)

        mask_frac = patchify3d(masks, patch_size).mean(dim=2)
        student_ids, teacher_ids = view_sampler(mask_frac, args.num_samples)
        student_patches, student_coord = select_tokens(
            student_images, student_ids, coord_grid, patch_size
        )
        teacher_patches, teacher_coord = select_tokens(
            teacher_images, teacher_ids, coord_grid, patch_size
        )
        # teacher positions the student also sees get seen_weight, the others 1
        seen = (teacher_ids[..., :, None] == student_ids[..., None, :]).any(dim=-1)
        token_weights = torch.where(seen.flatten(0, 1), args.seen_weight, 1.0)
        return {
            "image": images,
            "mask": masks,
            "student_patches": student_patches,
            "student_coord": student_coord,
            "teacher_patches": teacher_patches,
            "teacher_coord": teacher_coord,
            "token_weights": token_weights,
        }

    return fn


def count_batch_flops(
    args: DictConfig, model: nn.Module, view_sampler: Callable, device: torch.device
) -> float:
    # one training step on dummy images where every patch is in the mask. on a copy, so buffers
    # like the teacher center are untouched
    model = copy.deepcopy(model)
    amp_dtype = getattr(torch, args.amp_dtype)
    images = torch.zeros(args.batch_size, *args.grid_size, device=device)
    masks = torch.ones(args.batch_size, *args.grid_size, device=device)
    mask_frac = patchify3d(masks, args.patch_size).mean(dim=2)
    student_ids, teacher_ids = view_sampler(mask_frac, args.num_samples)
    coord_grid = model.encoder.coord_grid
    student_patches, student_coord = select_tokens(images, student_ids, coord_grid, args.patch_size)
    teacher_patches, teacher_coord = select_tokens(images, teacher_ids, coord_grid, args.patch_size)
    token_weights = torch.ones(teacher_patches.shape[:2], device=device)

    flop_counter = FlopCounterMode(display=False)
    with flop_counter:
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=args.amp):
            targets, _ = model.forward_teacher(teacher_patches, teacher_coord)
            loss, _ = model.forward_student(
                student_patches, student_coord, teacher_coord, targets, token_weights
            )
        loss.backward()
    return flop_counter.get_total_flops()


def plot_features(
    args: DictConfig,
    model: nn.Module,
    images: Tensor,
    masks: Tensor,
    step: int,
    amp_dtype: torch.dtype,
) -> Path:
    """Joint pca of teacher encoder features over all brain patches of a few images."""
    encoder = model.teacher_encoder
    patch_size = args.patch_size
    patch_grid_size = [size // patch_size for size in args.grid_size]
    images = images[: args.plot_num_images]
    masks = masks[: args.plot_num_images]

    # eager, the plot shapes differ from training and compiling them isn't worth it
    features = []
    in_masks = []
    with (
        torch.no_grad(),
        torch.autocast(device_type=images.device.type, dtype=amp_dtype, enabled=args.amp),
    ):
        for image, mask in zip(images, masks):
            patches = patchify3d(image[None], patch_size)
            in_mask = (patchify3d(mask[None], patch_size) > 0).any(dim=2)[0]
            tokens = encoder(patches[:, in_mask], encoder.coord_grid[:, in_mask])
            features.append(tokens[0].float())
            in_masks.append(in_mask)

    # top 3 pcs of the pooled tokens, each scaled to [0, 1] by its 1-99 percentiles
    pooled = torch.cat(features)
    mean = pooled.mean(dim=0)
    _, _, components = torch.linalg.svd(pooled - mean, full_matrices=False)
    projections = (pooled - mean) @ components[:3].T
    low = projections.quantile(0.01, dim=0)
    high = projections.quantile(0.99, dim=0)
    colors = ((projections - low) / (high - low)).clamp(0, 1)

    pca_maps = []
    start = 0
    for in_mask in in_masks:
        dense = colors.new_zeros(len(in_mask), 3)
        dense[in_mask] = colors[start : start + int(in_mask.sum())]
        start += int(in_mask.sum())
        pca_maps.append(dense.reshape(*patch_grid_size, 3))

    fig = plot_feature_pca(
        images.float().cpu().numpy(),
        masks.cpu().numpy(),
        torch.stack(pca_maps).cpu().numpy(),
    )
    path = Path(args.output_dir) / "plots" / f"feature_pca_{step:07d}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


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
