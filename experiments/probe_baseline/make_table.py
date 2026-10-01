import json
from pathlib import Path

from nanobrain.eval.tasks import TASKS

OUTPUT_DIR = Path("experiments/probe_baseline/output")
MODELS = [
    "raw_voxel",
    "vits_random_init",
    "vits_run2_epoch10",
    "vits_run2_epoch20",
    "vits_run2_epoch30",
    "vits_run2_epoch40",
]
HEADLINE_METRICS = ["auroc", "r2", "dice"]

print("| task | metric, n | " + " | ".join(MODELS) + " |")
print("|---" * (len(MODELS) + 2) + "|")
for task in TASKS:
    cells = []
    n_samples = ""
    for model in MODELS:
        path = OUTPUT_DIR / model / f"{task}.json"
        if not path.exists():
            cells.append("")
            continue
        result = json.loads(path.read_text())
        metric = next(metric for metric in HEADLINE_METRICS if metric in result)
        low, high = result[f"{metric}_ci"]
        cells.append(f"{result[metric]:.3f} [{low:.2f}, {high:.2f}]")
        n_samples = f"{metric} {result['n_samples']}"
    print(f"| {task} | {n_samples} | " + " | ".join(cells) + " |")
