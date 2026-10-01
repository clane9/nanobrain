import argparse
import importlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from nanobrain.eval.tasks import TASKS
from nanobrain.utils.misc import git_sha

logger = logging.getLogger(__name__)


def load_model(ckpt_path: str | Path) -> nn.Module:
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    module_name, class_name = checkpoint["model_class"].split(":")
    model_class = getattr(importlib.import_module(module_name), class_name)
    model = model_class.from_config(checkpoint["args"])
    model.load_state_dict(checkpoint["model"])
    return model


def probe_eval(
    model: nn.Module,
    task: str,
    batch_size: int = 4,
    num_workers: int = 8,
    device: str = "cuda",
    amp: bool = True,
) -> dict[str, Any]:
    probe = TASKS[task]()
    return probe(model, batch_size=batch_size, num_workers=num_workers, device=device, amp=amp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--tasks", nargs="+", choices=list(TASKS), default=list(TASKS))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.ckpt_path, map_location="cpu", weights_only=True)
    model = load_model(args.ckpt_path).to(args.device)

    for task in args.tasks:
        logger.info(f"running {task}")
        start = time.perf_counter()
        result = probe_eval(
            model, task, args.batch_size, args.num_workers, args.device, not args.no_amp
        )
        result = {
            "task": task,
            "ckpt_path": str(args.ckpt_path),
            "model_class": checkpoint["model_class"],
            "ckpt_git_sha": checkpoint["git_sha"],
            "eval_git_sha": git_sha(),
            "total_seconds": time.perf_counter() - start,
            **result,
        }
        with (args.output_dir / f"{task}.json").open("w") as f:
            json.dump(result, f)
        logger.info(f"{task} done in {result['total_seconds']:.0f}s")
