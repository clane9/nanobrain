import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

NUM_THREADS = 32


def load_record(meta_path: Path) -> dict:
    record = json.loads(meta_path.read_text())

    image_path = meta_path.with_name(meta_path.name.replace(".meta.json", ".npz"))
    record["path"] = str(image_path)

    for quantile, value in record.pop("qs").items():
        record[f"q{quantile}"] = value

    parts = record["name"].split("/")
    record["dataset"] = parts[1]
    record["subject"] = f"{parts[1]}/{parts[2]}"
    record["session"] = f"{parts[1]}/{parts[2]}/{parts[3]}"
    record["suffix"] = record["name"].removesuffix(".nii.gz").split("_")[-1]
    return record


def main(args: argparse.Namespace):
    meta_paths = []
    for dirpath, _, filenames in os.walk(args.root):
        for filename in filenames:
            if filename.endswith(".meta.json"):
                meta_paths.append(Path(dirpath) / filename)
    print(f"found {len(meta_paths)} metadata files in {args.root}")

    with ThreadPoolExecutor(NUM_THREADS) as executor:
        records = list(executor.map(load_record, meta_paths))

    df = pd.DataFrame.from_records(records)
    df = df.sort_values("name").reset_index(drop=True)

    out_path = args.root / "metadata.parquet"
    df.to_parquet(out_path)
    print(f"wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/FOMO300K_images"))
    args = parser.parse_args()
    main(args)
