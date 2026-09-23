from pathlib import Path

import numpy as np
import pandas as pd

FILELIST = Path("src/nanobrain/config/fomo300k_filelist_openneuro.txt")
OUT_PATH = Path("src/nanobrain/config/fomo300k_filelist_openneuro_subsampled.txt")
MAX_SUBJECTS_PER_DATASET = 200
SEED = 0
COLUMNS = ["collection", "dataset", "subject", "session"]


def main():
    paths = FILELIST.read_text().split()
    sessions = pd.DataFrame([p.removesuffix(".zip").split("/") for p in paths], columns=COLUMNS)
    sessions["path"] = paths
    sessions = sessions.sort_values("path").reset_index(drop=True)

    label_lengths = set(sessions.session.str.len())
    assert label_lengths == {6}, f"unexpected session labels {label_lengths}, sort may be wrong"

    first_session = sessions.groupby(["dataset", "subject"]).head(1)

    rng = np.random.default_rng(SEED)
    capped = []
    for dataset, group in first_session.groupby("dataset"):
        subjects = np.sort(group.subject.unique())
        if len(subjects) > MAX_SUBJECTS_PER_DATASET:
            subjects = rng.choice(subjects, MAX_SUBJECTS_PER_DATASET, replace=False)
        capped.append(group[group.subject.isin(subjects)])
    subsampled = pd.concat(capped).sort_values("path").reset_index(drop=True)

    stages = {"all": sessions, "first session": first_session, "capped": subsampled}
    for label, df in stages.items():
        subject_counts = df.groupby("dataset").subject.nunique().sort_values(ascending=False)
        top_10_share = subject_counts.iloc[:10].sum() / subject_counts.sum()
        print(
            f"{label}: {len(df):,} sessions, "
            f"{df.groupby(['dataset', 'subject']).ngroups:,} subjects, "
            f"{df.dataset.nunique()} datasets, top 10 datasets {top_10_share:.0%} of subjects"
        )

    OUT_PATH.write_text("\n".join(subsampled.path) + "\n")
    print(f"\nwrote {len(subsampled):,} paths to {OUT_PATH}")


if __name__ == "__main__":
    main()
