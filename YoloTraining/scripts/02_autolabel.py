"""Auto-label apples with Grounding DINO (open-vocabulary detector) on the local GPU.

Writes YOLO-format labels (class 0 = apple) to data/labels/<stem>.txt and, with
--preview, annotated images to data/preview/ for a visual sanity check.

    uv run scripts/02_autolabel.py --preview
"""
import argparse
from pathlib import Path

import cv2
import torch
from PIL import Image
from torchvision.ops import nms
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, default=ROOT / "data/frames")
    ap.add_argument("--labels", type=Path, default=ROOT / "data/labels")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-base")
    ap.add_argument("--prompt", default="apple. basket. box. leaf. stone. shoe. car.",
                    help="first phrase is the target; the rest soak up look-alike false positives")
    ap.add_argument("--box-thr", type=float, default=0.27)
    ap.add_argument("--text-thr", type=float, default=0.25)
    ap.add_argument("--max-area", type=float, default=0.08, help="drop boxes larger than this frac of the image")
    ap.add_argument("--min-side", type=int, default=6, help="drop boxes smaller than this (px)")
    ap.add_argument("--max-aspect", type=float, default=2.2, help="apples are roughly round")
    ap.add_argument("--min-bottom", type=float, default=0.25,
                    help="box bottom must be below this frac of image height (apples lie on the ground)")
    ap.add_argument("--tiles", type=int, default=3, help="zoomed tiles across the ground band (0 = full frame only)")
    ap.add_argument("--band", type=float, nargs=2, default=(0.2, 0.7),
                    help="vertical extent of the tiled ground band, as fractions of image height")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--redo", action="store_true", help="relabel frames that already have a label file")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model).to(dev).eval()

    args.labels.mkdir(parents=True, exist_ok=True)
    prev_dir = ROOT / "data/preview"
    if args.preview:
        prev_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted(args.frames.glob("*.jpg"))
    if args.limit:
        imgs = imgs[:: max(1, len(imgs) // args.limit)][: args.limit]
    if not args.redo:
        imgs = [p for p in imgs if not (args.labels / f"{p.stem}.txt").exists()]
    target = args.prompt.split(".")[0].strip()

    def detect(crops):
        """Run Grounding DINO on a batch of PIL crops -> list of (boxes xyxy in crop px, scores)."""
        inputs = proc(images=crops, text=[args.prompt] * len(crops), return_tensors="pt").to(dev)
        with torch.no_grad(), torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
            out = model(**inputs)
        results = proc.post_process_grounded_object_detection(
            out, inputs.input_ids, threshold=args.box_thr, text_threshold=args.text_thr,
            target_sizes=[(c.height, c.width) for c in crops],
        )
        dets = []
        for res in results:
            b, s = res["boxes"].float().cpu(), res["scores"].float().cpu()
            if len(b):
                labels = res.get("text_labels", res.get("labels"))
                m = torch.tensor([target in str(l) for l in labels], dtype=torch.bool)
                b, s = b[m], s[m]
            dets.append((b, s))
        return dets

    n_boxes = 0
    for p in tqdm(imgs, desc="labeling"):
        img = Image.open(p).convert("RGB")
        W, H = img.size
        boxes, scores = detect([img])[0]
        if args.tiles:
            # Zoomed crops of the ground band so distant (10-20 px) apples get enough resolution.
            y0, y1 = int(args.band[0] * H), int(args.band[1] * H)
            tw = int(W / args.tiles * 1.4)
            xs = [round(i * (W - tw) / (args.tiles - 1)) for i in range(args.tiles)]
            tile_dets = detect([img.crop((x, y0, x + tw, y1)) for x in xs])
            for x, (b, s) in zip(xs, tile_dets):
                if not len(b):
                    continue
                # drop boxes clipped by an interior tile edge (the full frame / neighbour tile has them whole)
                e = 3
                ok = ((b[:, 0] > e) | (x == 0)) & ((b[:, 2] < tw - e) | (x + tw >= W)) & (b[:, 1] > e)
                ok &= (b[:, 3] < (y1 - y0) - e) | (y1 >= H)
                b = b[ok] + torch.tensor([x, y0, x, y0], dtype=b.dtype)
                boxes, scores = torch.cat([boxes, b]), torch.cat([scores, s[ok]])
        if len(boxes):
            keep = nms(boxes, scores, 0.5)
            boxes, scores = boxes[keep], scores[keep]
            # drop "group" boxes drawn around a row of apples: they contain >= 2 other boxes' centres
            c = (boxes[:, :2] + boxes[:, 2:]) / 2
            inside = ((c[None, :, 0] > boxes[:, None, 0]) & (c[None, :, 0] < boxes[:, None, 2])
                      & (c[None, :, 1] > boxes[:, None, 1]) & (c[None, :, 1] < boxes[:, None, 3]))
            inside.fill_diagonal_(False)
            keep = inside.sum(1) < 2
            boxes, scores = boxes[keep], scores[keep]

        lines, draw = [], []
        for (x1, y1, x2, y2), s in zip(boxes.tolist(), scores.tolist()):
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
            w, h = x2 - x1, y2 - y1
            if min(w, h) < args.min_side or w * h > args.max_area * W * H:
                continue
            if max(w / h, h / w) > args.max_aspect or y2 < args.min_bottom * H:
                continue
            lines.append(f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {w / W:.6f} {h / H:.6f}")
            draw.append((int(x1), int(y1), int(x2), int(y2), s))
        (args.labels / f"{p.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        n_boxes += len(lines)

        if args.preview:
            im = cv2.imread(str(p))
            for x1, y1, x2, y2, s in draw:
                cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(im, f"{s:.2f}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imwrite(str(prev_dir / p.name), im, [cv2.IMWRITE_JPEG_QUALITY, 80])

    print(f"{len(imgs)} images, {n_boxes} boxes ({n_boxes / max(1, len(imgs)):.1f}/img) -> {args.labels}")


if __name__ == "__main__":
    main()
