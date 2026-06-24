"""Standalone RGB → policy-mask pipeline + lane curve fitting from blue mask.

Reads raw_{n}.png from /tmp/mushr_frames/, applies the same
HSV-blue-mask → morphological opening → downsample pipeline used in the
RL training loop, then fits two curves to the lane markings (detected from
the blue mask via per-row pixel clustering) and overlays them on a debug image.

Usage:
    python preprocess_raw.py
    python preprocess_raw.py --in_dir /tmp/mushr_frames --out_dir /tmp/mushr_frames
"""

import argparse
import os
import re

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# ── Parameters (mirror source/tasks/cone_track/parameters.py) ──────────
CAM_W: int = 848
CAM_H: int = 480
VISION_CROP_TOP_FRACTION: float = 0.5
VISION_CROP_BOTTOM_FRACTION: float = 0.0
DOWNSAMPLING_AMOUNT: int = 1

# Derived
_remaining = max(0.0, 1.0 - VISION_CROP_TOP_FRACTION - VISION_CROP_BOTTOM_FRACTION)
POLICY_H = max(1, int(CAM_H * _remaining) // DOWNSAMPLING_AMOUNT)
POLICY_W = CAM_W // DOWNSAMPLING_AMOUNT

# ── Tunable lane curve-fitting parameters ────────────────────────────
# Region of interest: skip the very top (sky/horizon) and car body
ROI_ROW_START: int = 30
ROI_ROW_END: int = 235
# Minimum tracked points to fit a lane curve
MIN_LANE_POINTS: int = 10
# Blue mask threshold for "this pixel is lane"
BLUE_THRESHOLD: float = 0.3
# Max pixel change per row during bottom-up lane tracking
MAX_STEP: float = 200.0
# When a tracker has no data, drift toward center at this rate (px/row)
DRIFT_RATE: float = 8.0
# Minimum connected-component size to keep (removes noise blobs)
MIN_BLOB_SIZE: int = 50


def process_single_rgb(rgb: np.ndarray) -> np.ndarray:
    """Standalone version of ConeTrackEnv._process_single_rgb.

    Args:
        rgb: [H, W, 3] uint8 RGB image.

    Returns:
        [POLICY_H, POLICY_W] float32 mask in [0, 1].
    """
    rgb_t = torch.from_numpy(rgb.astype(np.float32) / 255.0)

    r, g, b = rgb_t[..., 0], rgb_t[..., 1], rgb_t[..., 2]
    cmax, _ = rgb_t.max(dim=-1)
    cmin, _ = rgb_t.min(dim=-1)
    delta = cmax - cmin

    h = torch.zeros_like(cmax)
    valid = delta > 1e-8
    mask_r = (cmax == r) & valid
    h[mask_r] = ((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6.0 * 60.0
    mask_g = (cmax == g) & valid & ~mask_r
    h[mask_g] = ((b[mask_g] - r[mask_g]) / delta[mask_g] + 2.0) * 60.0
    mask_b = valid & ~mask_r & ~mask_g
    h[mask_b] = ((r[mask_b] - g[mask_b]) / delta[mask_b] + 4.0) * 60.0
    h = h % 360.0
    s = torch.where(cmax > 0, delta / cmax, torch.zeros_like(cmax))
    v = cmax

    in_h_low, in_h_high = 180.0, 300.0
    in_s_low = 30.0 / 255.0
    in_v_low = 50.0 / 255.0

    mask = ((h >= in_h_low) & (h <= in_h_high) & (s >= in_s_low) & (v >= in_v_low)).float()

    k = 3
    p = k // 2
    mask = -F.max_pool2d(-mask.unsqueeze(0).unsqueeze(0), k, stride=1, padding=p)
    mask = F.max_pool2d(mask, k, stride=1, padding=p)
    mask = mask.squeeze()

    mask_4d = F.interpolate(
        mask.unsqueeze(0).unsqueeze(0),
        size=(POLICY_H, POLICY_W),
        mode="bilinear",
        align_corners=False,
    )
    return mask_4d.squeeze().numpy()


def find_lanes_from_mask(mask: np.ndarray) -> tuple:
    """Detect left/right lane boundaries via bottom-up cluster tracking.

    For each row of the mask, finds clusters of blue pixels (lane markings).
    Tracks them bottom-to-top so that a right lane crossing to the left half
    of the image (common on curves) is followed correctly.

    Returns:
        (left_xs, right_xs): arrays of x for each row, or None.
    """
    h, w = mask.shape
    mid = w // 2

    # ── Clean up mask before tracking ────────────────────────────────────
    binary = (mask > BLUE_THRESHOLD).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    clean = np.zeros_like(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_BLOB_SIZE:
            clean[labels == i] = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)

    left_pts = []
    right_pts = []
    l_tracker = None
    r_tracker = None

    for r in reversed(range(ROI_ROW_START, min(ROI_ROW_END, h))):
        row = clean[r, :]

        blue_idx = np.where(row > 0)[0]
        if len(blue_idx) < 2:
            if l_tracker is not None:
                l_tracker += np.sign(mid - l_tracker) * DRIFT_RATE
            if r_tracker is not None:
                r_tracker += np.sign(mid - r_tracker) * DRIFT_RATE
            continue

        # ── Cluster adjacent blue pixels ────────────────────────────────
        clusters = []
        current = [blue_idx[0]]
        for x in blue_idx[1:]:
            if x - current[-1] <= 5:
                current.append(x)
            else:
                if len(current) >= 2:
                    clusters.append(np.median(current))
                current = [x]
        if len(current) >= 2:
            clusters.append(np.median(current))

        if not clusters:
            continue

        clusters = np.array(clusters)
        left_clusters = clusters[clusters <= mid]
        right_clusters = clusters[clusters > mid]

        # ── Initialize trackers on first row with data ──────────────────
        if l_tracker is None and r_tracker is None:
            if len(left_clusters) > 0:
                l_tracker = left_clusters[np.argmax(left_clusters)]
                left_pts.append((r, l_tracker))
            if len(right_clusters) > 0:
                r_tracker = right_clusters[np.argmin(right_clusters)]
                right_pts.append((r, r_tracker))
            continue

        # ── Each tracker picks the closest cluster on its own side ──────
        l_before = l_tracker
        r_before = r_tracker

        if l_tracker is not None and len(left_clusters) > 0:
            dists = np.abs(left_clusters - l_tracker)
            best = np.argmin(dists)
            if dists[best] <= MAX_STEP:
                left_pts.append((r, left_clusters[best]))
                l_tracker = left_clusters[best]

        if r_tracker is not None and len(right_clusters) > 0:
            dists = np.abs(right_clusters - r_tracker)
            best = np.argmin(dists)
            if dists[best] <= MAX_STEP:
                right_pts.append((r, right_clusters[best]))
                r_tracker = right_clusters[best]

        # Initialize missing tracker if clusters appear on its side
        if l_tracker is None and len(left_clusters) > 0:
            l_tracker = left_clusters[np.argmax(left_clusters)]
            left_pts.append((r, l_tracker))
        if r_tracker is None and len(right_clusters) > 0:
            r_tracker = right_clusters[np.argmin(right_clusters)]
            right_pts.append((r, r_tracker))

        # Drift trackers that didn't receive a cluster this row
        if l_tracker is not None and l_tracker == l_before:
            l_tracker += np.sign(mid - l_tracker) * DRIFT_RATE
        if r_tracker is not None and r_tracker == r_before:
            r_tracker += np.sign(mid - r_tracker) * DRIFT_RATE

    # ── Fit a smooth polynomial for each lane ───────────────────────────
    def fit_lane(pts) -> np.ndarray | None:
        if len(pts) < MIN_LANE_POINTS:
            return None
        pts = np.array(pts)
        rows, cols = pts[:, 0], pts[:, 1]

        # Iterative MAD outlier rejection
        for _ in range(1):
            if len(rows) < MIN_LANE_POINTS:
                return None
            try:
                coeffs = np.polyfit(rows, cols, 3)
            except np.linalg.LinAlgError:
                try:
                    coeffs = np.polyfit(rows, cols, 2)
                except np.linalg.LinAlgError:
                    try:
                        coeffs = np.polyfit(rows, cols, 1)
                    except np.linalg.LinAlgError:
                        return None
            pred = np.polyval(coeffs, rows)
            resid = np.abs(cols - pred)
            mad = np.median(resid)
            if mad < 0.5:
                mad = 0.5
            keep = resid < 3.0 * mad
            if keep.sum() >= MIN_LANE_POINTS:
                rows = rows[keep]
                cols = cols[keep]
            else:
                break

        try:
            coeffs = np.polyfit(rows, cols, 3)
        except np.linalg.LinAlgError:
            try:
                coeffs = np.polyfit(rows, cols, 2)
            except np.linalg.LinAlgError:
                try:
                    coeffs = np.polyfit(rows, cols, 1)
                except np.linalg.LinAlgError:
                    return None

        fitted = np.full(h, np.nan)
        r_min, r_max = int(rows.min()), int(rows.max())
        eval_rows = np.arange(r_min, r_max + 1)
        vals = np.polyval(coeffs, eval_rows)
        vals = np.clip(vals, 0, w - 1)
        fitted[r_min:r_max + 1] = vals
        return fitted

    left_fitted = fit_lane(left_pts)
    right_fitted = fit_lane(right_pts)

    return left_fitted, right_fitted, left_pts, right_pts


def fit_and_draw_lanes(debug: np.ndarray, left_xs: np.ndarray,
                       right_xs: np.ndarray) -> np.ndarray:
    """Draw lane curves on debug image."""
    h, w = debug.shape[:2]
    overlay = debug.copy()

    def draw_one(xs, color):
        if xs is None:
            return
        valid = ~np.isnan(xs)
        if valid.sum() < 2:
            return
        ys = np.where(valid)[0]
        pts = np.stack([xs[valid].astype(int), ys], axis=1)
        cv2.polylines(overlay, [pts], False, color, 3)

    draw_one(left_xs, (255, 0, 0))
    draw_one(right_xs, (0, 255, 0))

    return overlay


def draw_lane_points(debug: np.ndarray, left_pts: list,
                     right_pts: list) -> np.ndarray:
    """Draw tracked lane cluster points as dots on debug image."""
    overlay = debug.copy()
    for r, x in left_pts:
        cv2.circle(overlay, (int(x), int(r)), 2, (0, 255, 255), -1)
    for r, x in right_pts:
        cv2.circle(overlay, (int(x), int(r)), 2, (255, 255, 0), -1)
    return overlay


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce the RL policy vision pipeline on saved raw frames."
    )
    parser.add_argument(
        "--in_dir",
        default="/tmp/mushr_frames",
        help="Directory with raw_{n}.png frames.",
    )
    parser.add_argument(
        "--out_dir",
        default="/tmp/mushr_frames",
        help="Where to write policy_{n}.png masks (default: same as --in_dir).",
    )
    args = parser.parse_args()

    in_dir = args.in_dir
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    raw_files = sorted(f for f in os.listdir(in_dir) if re.match(r"raw_\d+\.png$", f))
    if not raw_files:
        print(f"[ERROR] No raw_*.png files found in {in_dir}")
        return

    print(f"Found {len(raw_files)} raw images in {in_dir}")
    print(f"Policy resolution: {POLICY_W}×{POLICY_H}")

    for fname in raw_files:
        counter = re.search(r"raw_(\d+)\.png", fname).group(1)

        raw_path = os.path.join(in_dir, fname)
        bgr = cv2.imread(raw_path)
        if bgr is None:
            print(f"  SKIP  {fname}  (could not read)")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Run the same pipeline as the RL training loop
        mask = process_single_rgb(rgb)

        # Save as grayscale PNG
        mask_uint8 = (mask * 255).astype(np.uint8)
        out_path = os.path.join(out_dir, f"policy_{counter}.png")
        cv2.imwrite(out_path, mask_uint8)

        coverage = mask.mean() * 100
        print(f"  {fname} → policy_{counter}.png  "
              f"({mask.shape[1]}×{mask.shape[0]}, "
              f"blue coverage {coverage:.1f}%)")

        # ── Lane curve fitting on the blue mask (NOT grayscale) ──────
        left_xs, right_xs, left_pts, right_pts = find_lanes_from_mask(mask)

        # Draw curves on the original BGR image
        debug = fit_and_draw_lanes(bgr, left_xs, right_xs)
        lanes_path = os.path.join(out_dir, f"lanes_{counter}.png")
        cv2.imwrite(lanes_path, debug)

        # Draw curves + tracked points on the mask
        mask_bgr = cv2.cvtColor(mask_uint8, cv2.COLOR_GRAY2BGR)
        debug_mask = fit_and_draw_lanes(mask_bgr, left_xs, right_xs)
        debug_mask = draw_lane_points(debug_mask, left_pts, right_pts)
        mask_lanes_path = os.path.join(out_dir, f"lanes_mask_{counter}.png")
        cv2.imwrite(mask_lanes_path, debug_mask)

        left_ok = left_xs is not None
        right_ok = right_xs is not None
        print(f"           → lanes_{counter}.png / lanes_mask_{counter}.png  "
              f"({'L' if left_ok else '-'}{'R' if right_ok else '-'} curves, "
              f"{len(left_pts)}L/{len(right_pts)}R pts)")

    print("Done.")


if __name__ == "__main__":
    main()
