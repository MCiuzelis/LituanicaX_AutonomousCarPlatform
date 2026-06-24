"""GPU lane feature extraction from binary blue-mask images.

All operations are fully batched across environments — no Python per-env loop.
"""

import torch


def _find_lane_boundaries(mask_binary: torch.Tensor, center_frac: float = 0.5):
    """Find left and right lane boundaries for each row.

    For each row in each batch element, the left lane boundary is the
    rightmost blue pixel left of the image centre, and the right lane
    boundary is the leftmost blue pixel at or right of the image centre.

    All operations are fully parallel GPU reductions — no cumsum/argmax.

    Args:
        mask_binary: [N, H, W] bool tensor – thresholded mask.
        center_frac: fraction of width used as the left/right split.

    Returns:
        left_bnd:  [N, H] float32 – column of rightmost left-side blue pixel,
                   NaN where none exists.
        right_bnd: [N, H] float32 – column of leftmost right-side blue pixel,
                   NaN where none exists.
    """
    N, H, W = mask_binary.shape
    device = mask_binary.device

    split_col = int(W * center_frac)

    col_grid = torch.arange(W, device=device).float()

    left_mask = mask_binary & (col_grid < split_col)
    right_mask = mask_binary & (col_grid >= split_col)

    # Left boundary: rightmost blue pixel on left side (max col index)
    left_cols = torch.where(left_mask, col_grid, -1.0)
    left_bnd = left_cols.max(dim=-1).values
    left_bnd[left_bnd < 0] = torch.nan

    # Right boundary: leftmost blue pixel on right side (min col index)
    right_cols = torch.where(right_mask, col_grid, float('inf'))
    right_bnd = right_cols.min(dim=-1).values
    right_bnd[~torch.isfinite(right_bnd)] = torch.nan

    return left_bnd, right_bnd


def _batched_weighted_polyfit(y: torch.Tensor, x: torch.Tensor,
                               w: torch.Tensor, deg: int = 3):
    """Fully batched weighted polynomial fit  x = f(y)  for each batch element.

    Uses a common Vandermonde matrix and per-batch weight vectors so the
    solve is a single  [N, 4, 4]  batched linear system — no Python loop.

    Args:
        y: [N, P]  y-coordinates (same grid for all batches, normalised).
        x: [N, P]  x-coordinates, NaN where lane is missing for that row.
        w: [N, P]  weights (0 for NaN rows, 1 for valid rows).

    Returns:
        coeffs: [N, deg+1]  polynomial coefficients [a0, a1, ..., a_deg].
    """
    N, P = y.shape
    device = y.device

    # Vandermonde matrix for the common y-grid  [1, P, deg+1]
    terms = [(y ** (deg - i)).unsqueeze(-1) for i in range(deg + 1)]
    V = torch.cat(terms, dim=-1)  # [N, P, deg+1]

    # Handle NaN x-values by setting both weight and x to 0 for those entries
    x_clean = torch.where(w > 0, x, torch.zeros_like(x))

    # Weighted normal equations  X^T W X  and  X^T W y
    WV = V * w.unsqueeze(-1)  # [N, P, 4]
    VtWV = torch.matmul(V.transpose(-1, -2), WV)  # [N, 4, 4]
    VtWy = (WV * x_clean.unsqueeze(-1)).sum(dim=1)  # [N, 4]

    eye = torch.eye(deg + 1, device=device).unsqueeze(0)  # [1, 4, 4]
    VtWV = VtWV + 1e-6 * eye  # regularisation for stability

    solved = torch.linalg.solve(VtWV, VtWy.unsqueeze(-1))  # [N, 4, 1]
    return solved.squeeze(-1)  # [N, 4]


@torch.compile
def extract_lane_features(mask_binary: torch.Tensor) -> torch.Tensor:
    """Extract 18 lane-geometry parameters from a binary blue mask.

    All operations are fully batched — no Python per-env loop.

    Args:
        mask_binary: [N, H, W] float32 tensor, values in {0, 1}.

    Returns:
        feats: [N, 18] float32 tensor with columns:

            [0:4]    left  polynomial coeffs   [a0, a1, a2, a3]
            [4:8]    right polynomial coeffs   [a0, a1, a2, a3]
            [8]      left  lane x at bottom    (normalised by W)
            [9]      right lane x at bottom    (normalised by W)
            [10]     lane width at bottom      (right_x - left_x)
            [11]     centre offset             ((l+r)/2 - 0.5)
            [12]     left  curvature at bottom (2*a2 + 6*a3)
            [13]     right curvature at bottom
            [14]     left  detected flag       {0, 1}
            [15]     right detected flag       {0, 1}
            [16]     left  confidence          (blue fraction in left half)
            [17]     right confidence          (blue fraction in right half)
    """
    N, H, W = mask_binary.shape
    device = mask_binary.device

    binary = mask_binary > 0.5

    # ── Per-row lane boundaries  [N, H] ───────────────────────────────
    left_bnd, right_bnd = _find_lane_boundaries(binary)

    # Crop away top 20 % where lanes are too small to be reliable
    scan_start = int(H * 0.2)
    H_eff = H - scan_start
    y_norm = torch.arange(H, device=device).float() / H  # [H]

    l_bnd = left_bnd[:, scan_start:]   # [N, H_eff]
    r_bnd = right_bnd[:, scan_start:]  # [N, H_eff]
    y_grid = y_norm[scan_start:].unsqueeze(0).expand(N, -1)  # [N, H_eff]

    # ── Feature tensor ────────────────────────────────────────────────
    feats = torch.zeros(N, 18, device=device, dtype=torch.float32)

    # ── Confidence (fully batched) ────────────────────────────────────
    W2 = W // 2
    feats[:, 16] = binary[:, :, :W2].float().mean(dim=(1, 2))
    feats[:, 17] = binary[:, :, W2:].float().mean(dim=(1, 2))

    # ── Validity masks ────────────────────────────────────────────────
    l_valid = ~torch.isnan(l_bnd)  # [N, H_eff]
    r_valid = ~torch.isnan(r_bnd)
    l_any = l_valid.any(dim=-1)   # [N]
    r_any = r_valid.any(dim=-1)

    # ── Fallback bottom-x (batched) ───────────────────────────────────
    bottom_left = l_bnd[:, -5:].nan_to_num(nan=W // 2).mean(dim=-1) / W
    bottom_right = r_bnd[:, -5:].nan_to_num(nan=W // 2).mean(dim=-1) / W
    feats[:, 8] = torch.where(l_any, bottom_left, torch.full_like(bottom_left, 0.5))
    feats[:, 9] = torch.where(r_any, bottom_right, torch.full_like(bottom_right, 0.5))
    feats[:, 14] = l_any.float()
    feats[:, 15] = r_any.float()

    # ── Batched polynomial fitting ────────────────────────────────────
    l_x = l_bnd / W  # [N, H_eff]
    r_x = r_bnd / W

    l_enough = l_valid.sum(dim=-1) >= 4  # [N]
    r_enough = r_valid.sum(dim=-1) >= 4

    l_w = l_valid.float()
    r_w = r_valid.float()

    # Only fit when there are ≥4 points; otherwise keep coeffs = 0
    if l_enough.any():
        l_coeffs = _batched_weighted_polyfit(y_grid, l_x, l_w)
        feats[l_enough, 0:4] = l_coeffs[l_enough]
        feats[l_enough, 8] = (l_coeffs[l_enough, 0] + l_coeffs[l_enough, 1]
                              + l_coeffs[l_enough, 2] + l_coeffs[l_enough, 3])
        feats[l_enough, 12] = 2.0 * l_coeffs[l_enough, 2] + 6.0 * l_coeffs[l_enough, 3]
        feats[l_enough, 14] = 1.0

    if r_enough.any():
        r_coeffs = _batched_weighted_polyfit(y_grid, r_x, r_w)
        feats[r_enough, 4:8] = r_coeffs[r_enough]
        feats[r_enough, 9] = (r_coeffs[r_enough, 0] + r_coeffs[r_enough, 1]
                              + r_coeffs[r_enough, 2] + r_coeffs[r_enough, 3])
        feats[r_enough, 13] = 2.0 * r_coeffs[r_enough, 2] + 6.0 * r_coeffs[r_enough, 3]
        feats[r_enough, 15] = 1.0

    # ── Derived features (fully batched) ──────────────────────────────
    both = l_any & r_any
    only_left = l_any & ~r_any
    only_right = ~l_any & r_any

    feats[both, 10] = feats[both, 9] - feats[both, 8]
    feats[both, 11] = (feats[both, 8] + feats[both, 9]) / 2.0 - 0.5
    feats[only_left, 11] = feats[only_left, 8] - 0.5
    feats[only_right, 11] = feats[only_right, 9] - 0.5

    return feats
