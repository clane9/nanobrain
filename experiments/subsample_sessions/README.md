# Subsample sessions

Freeze the pretraining split up front: first session per subject, then cap subjects per dataset.
Output is committed to `src/nanobrain/config/fomo300k_filelist_openneuro_subsampled.txt` and is the
input to `prepare`.

```bash
uv run python experiments/subsample_sessions/subsample_sessions.py
```

| Stage | Sessions | Subjects | Top 10 datasets |
|---|---|---|---|
| All openneuro | 45,376 | 37,948 | 18% |
| First session per subject | 37,948 | 37,948 | 18% |
| ≤200 subjects per dataset (seed 0) | 32,809 | 32,809 | 6% |

Picking at most one image per suffix stays in `experiments/filter_images/`; it needs the prepared
metadata.
