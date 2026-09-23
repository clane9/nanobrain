import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

from filter_images import RULES

VIEWS = ["sag", "cor", "ax"]
THUMB_SIZE = 96
LABEL_HEIGHT = 12
COLUMNS = 6
ROWS = 10


def make_montage(rows: pd.DataFrame) -> Image.Image:
    num_rows = (len(rows) + COLUMNS - 1) // COLUMNS
    cell_width = THUMB_SIZE * len(VIEWS)
    cell_height = THUMB_SIZE + LABEL_HEIGHT
    montage = Image.new("L", (COLUMNS * cell_width, num_rows * cell_height), 255)
    draw = ImageDraw.Draw(montage)

    for ii, row in enumerate(rows.itertuples()):
        x = (ii % COLUMNS) * cell_width
        y = (ii // COLUMNS) * cell_height
        draw.text((x + 2, y), f"{row.dataset} {row.suffix}", fill=0)
        stem = row.path.removesuffix(".npz")
        for jj, view in enumerate(VIEWS):
            thumb = Image.open(f"{stem}.{view}.jpg").resize((THUMB_SIZE, THUMB_SIZE))
            montage.paste(thumb, (x + jj * THUMB_SIZE, y + LABEL_HEIGHT))
    return montage


def main(args: argparse.Namespace):
    filters = pd.read_parquet(args.filters)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    page_size = COLUMNS * ROWS

    for rule in RULES:
        excluded = filters[filters[rule]]
        sample = excluded.sample(min(page_size, len(excluded)), random_state=args.seed)
        make_montage(sample).save(args.out_dir / f"excluded_{rule}.png")

    kept = filters[filters.keep]
    sample = kept.sample(args.num_kept, random_state=args.seed)
    for page, start in enumerate(range(0, args.num_kept, page_size)):
        rows = sample.iloc[start : start + page_size]
        make_montage(rows).save(args.out_dir / f"kept_{page}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filters", type=Path, default=Path("experiments/filter_images/filter.parquet")
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("experiments/filter_images/figures/montages")
    )
    parser.add_argument("--num-kept", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)
