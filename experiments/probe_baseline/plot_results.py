import json
from pathlib import Path

import matplotlib.pyplot as plt

from nanobrain.eval.tasks import HEADLINE_METRICS, TASKS

OUTPUT_DIR = Path("output")
FIGURE_DIR = Path("figures")
MODELS = [
    "raw_voxel",
    "vits_random_init",
    "vits_run2_epoch10",
    "vits_run2_epoch20",
    "vits_run2_epoch30",
    "vits_run2_epoch40",
]
MODEL_LABELS = ["raw", "rand", "ep10", "ep20", "ep30", "ep40"]

FIGURE_DIR.mkdir(exist_ok=True)
n_rows = (len(TASKS) + 3) // 4
fig, axes = plt.subplots(n_rows, 4, figsize=(16, 3.3 * n_rows))
for ax in axes.flat[len(TASKS) :]:
    ax.axis("off")
for ax, task in zip(axes.flat, TASKS):
    for ii, model in enumerate(MODELS):
        result = json.loads((OUTPUT_DIR / model / f"{task}.json").read_text())
        metric = HEADLINE_METRICS[task]
        low, high = result[f"{metric}_ci"]
        value = result[metric]
        ax.errorbar(ii, value, yerr=[[value - low], [high - value]], fmt="o", capsize=3)
    ax.set_title(task.removeprefix("fomo_"))
    ax.set_ylabel(f"{metric} (n={result['n_samples']})")
    ax.set_xticks(range(len(MODELS)), MODEL_LABELS)
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "results.png", dpi=100)
