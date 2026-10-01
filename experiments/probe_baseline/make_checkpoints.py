from pathlib import Path

import torch

from nanobrain.model import ViTMAE3D

RUN_DIR = Path("experiments/baseline/output/baseline/vits_run2")
RUN_GIT_SHA = "fe34e0d"
OUT_DIR = Path("experiments/probe_baseline/checkpoints")
RANDOM_INIT_SEED = 0

OUT_DIR.mkdir(parents=True, exist_ok=True)

for epoch in (9, 19, 29, 39):
    checkpoint = torch.load(
        RUN_DIR / f"checkpoint-{epoch:05d}.pth", map_location="cpu", weights_only=True
    )
    slim = {
        "model": checkpoint["model"],
        "args": checkpoint["args"],
        "model_class": "nanobrain.model:ViTMAE3D",
        "git_sha": RUN_GIT_SHA,
    }
    torch.save(slim, OUT_DIR / f"vits_run2_epoch{epoch + 1:02d}.pth")

torch.manual_seed(RANDOM_INIT_SEED)
model = ViTMAE3D.from_config(checkpoint["args"])
slim = {
    "model": model.state_dict(),
    "args": checkpoint["args"],
    "model_class": "nanobrain.model:ViTMAE3D",
    "git_sha": RUN_GIT_SHA,
}
torch.save(slim, OUT_DIR / "vits_random_init.pth")

slim = {
    "model": {},
    "args": {"patch_size": 8},
    "model_class": "nanobrain.eval.probe:RawVoxelEncoder",
    "git_sha": RUN_GIT_SHA,
}
torch.save(slim, OUT_DIR / "raw_voxel.pth")
