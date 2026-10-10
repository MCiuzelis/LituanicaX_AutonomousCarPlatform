"""Assemble the YOLO dataset from data/frames + data/labels.

Splits by *video* (not by frame) so near-identical neighbouring frames never end up on
both sides of the train/val split. Output: data/yolo/{images,labels}/{train,val} + data.yaml

    uv run scripts/03_build_dataset.py
"""
import argparse
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def link(src: Path, dst: Path):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, default=ROOT / "data/frames")
    ap.add_argument("--labels", type=Path, default=ROOT / "data/labels")
    ap.add_argument("--out", type=Path, default=ROOT / "data/yolo")
    ap.add_argument("--val-videos", nargs="+", default=["20261010_131416", "20261010_131458", "20261010_131840"],
                    help="video stems held out for validation")
    args = ap.parse_args()

    if args.out.exists():
        shutil.rmtree(args.out)
    for split in ("train", "val"):
        (args.out / "images" / split).mkdir(parents=True)
        (args.out / "labels" / split).mkdir(parents=True)

    counts = {"train": [0, 0, 0], "val": [0, 0, 0]}  # images, boxes, empty images
    for img in sorted(args.frames.glob("*.jpg")):
        lbl = args.labels / f"{img.stem}.txt"
        if not lbl.exists():
            continue
        split = "val" if any(img.stem.startswith(v) for v in args.val_videos) else "train"
        link(img, args.out / "images" / split / img.name)
        link(lbl, args.out / "labels" / split / lbl.name)
        n = sum(1 for line in lbl.read_text().splitlines() if line.strip())
        c = counts[split]
        c[0] += 1
        c[1] += n
        c[2] += n == 0

    # no `path:` key -> Ultralytics resolves train/val relative to this yaml, so the folder can move
    (args.out / "data.yaml").write_text("train: images/train\nval: images/val\nnames:\n  0: apple\n")
    for split, (ni, nb, ne) in counts.items():
        print(f"{split}: {ni} images, {nb} boxes, {ne} background-only")
    print(f"-> {args.out / 'data.yaml'}")


if __name__ == "__main__":
    main()
