import json
from pathlib import Path

import matplotlib.pyplot as plt

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
TASKS = [
    "fomo_task1_adc_lesion",
    "fomo_task2_dwi_tumor",
    "fomo_task3_t1w_hippocampus",
    "fomo_task4_t2w_nerve",
    "fomo_task4_t2w_vessel",
    "fomo_task5_t1w_cortex",
]
METRICS = ["dice", "average_precision", "voxel_auroc"]

FIGURE_DIR.mkdir(exist_ok=True)
fig, axes = plt.subplots(len(TASKS), len(METRICS), figsize=(12, 2.6 * len(TASKS)))
for row_axes, task in zip(axes, TASKS):
    for ax, metric in zip(row_axes, METRICS):
        for ii, model in enumerate(MODELS):
            result = json.loads((OUTPUT_DIR / model / f"{task}.json").read_text())
            low, high = result[f"{metric}_ci"]
            value = result[metric]
            ax.errorbar(ii, value, yerr=[[value - low], [high - value]], fmt="o", capsize=3)
        ax.set_title(f"{task.removeprefix('fomo_')} {metric}", fontsize=9)
        ax.set_xticks(range(len(MODELS)), MODEL_LABELS)
        ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "seg_metrics.png", dpi=80)
