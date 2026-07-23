from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple

import torch


def intrinsics_from_focal_center(
    fx: torch.Tensor | float,
    fy: torch.Tensor | float,
    cx: torch.Tensor | float,
    cy: torch.Tensor | float,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """
    Build a (normalized) 3x3 intrinsics matrix.

    MoGe uses a normalized camera convention where principal point is expressed
    in [0, 1] and focal is in "normalized" units (so that fov can be computed as
    `2 * atan(0.5 / f)`).
    """
    fx_t = torch.as_tensor(fx, dtype=dtype, device=device)
    inferred_device = fx_t.device if device is None else torch.device(device)
    inferred_dtype = fx_t.dtype if dtype is None else dtype

    fx_t = fx_t.to(device=inferred_device, dtype=inferred_dtype)
    fy_t = torch.as_tensor(fy, dtype=inferred_dtype, device=inferred_device)
    cx_t = torch.as_tensor(cx, dtype=inferred_dtype, device=inferred_device)
    cy_t = torch.as_tensor(cy, dtype=inferred_dtype, device=inferred_device)

    fx_t, fy_t, cx_t, cy_t = torch.broadcast_tensors(fx_t, fy_t, cx_t, cy_t)
    zeros = torch.zeros_like(fx_t)
    ones = torch.ones_like(fx_t)

    row0 = torch.stack([fx_t, zeros, cx_t], dim=-1)
    row1 = torch.stack([zeros, fy_t, cy_t], dim=-1)
    row2 = torch.stack([zeros, zeros, ones], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def image_uv(
    *,
    width: int,
    height: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """
    Normalized UV in [0, 1] at pixel centers, shape (H, W, 2), (u, v).
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid size: width={width}, height={height}")
    xs = (torch.arange(width, dtype=dtype, device=device) + 0.5) / float(width)
    ys = (torch.arange(height, dtype=dtype, device=device) + 0.5) / float(height)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([grid_x, grid_y], dim=-1)


def image_pixel_center(
    *,
    width: int,
    height: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """
    Pixel-center coordinates in pixel units, shape (H, W, 2), (u, v).
    """
    uv01 = image_uv(width=width, height=height, dtype=dtype, device=device)
    scale = torch.tensor([float(width), float(height)], dtype=dtype, device=device)
    return uv01 * scale


def depth_to_points(depth: torch.Tensor, *, intrinsics: torch.Tensor) -> torch.Tensor:
    """
    Unproject a depth map to a point map in camera coordinates.

    - depth: (B, H, W) or (B, 1, H, W)
    - intrinsics: (B, 3, 3) (normalized intrinsics)

    Returns: (B, H, W, 3)
    """
    if depth.dim() == 4 and depth.shape[1] == 1:
        depth = depth[:, 0]
    if depth.dim() != 3:
        raise ValueError(f"depth must be (B,H,W) or (B,1,H,W), got {tuple(depth.shape)}")
    if intrinsics.dim() != 3 or intrinsics.shape[-2:] != (3, 3):
        raise ValueError(f"intrinsics must be (B,3,3), got {tuple(intrinsics.shape)}")
    if intrinsics.shape[0] != depth.shape[0]:
        raise ValueError("Batch size mismatch between depth and intrinsics")

    b, h, w = depth.shape
    device = depth.device
    dtype = depth.dtype

    uv = image_uv(width=w, height=h, dtype=dtype, device=device)  # (H,W,2) in [0,1]
    ones = torch.ones((h, w, 1), dtype=dtype, device=device)
    pix = torch.cat([uv, ones], dim=-1)  # (H,W,3)
    pix = pix.unsqueeze(0).expand(b, -1, -1, -1)  # (B,H,W,3)

    k_inv = torch.inverse(intrinsics).to(dtype=dtype)
    rays = torch.einsum("bij,bhwj->bhwi", k_inv, pix)  # (B,H,W,3)
    return rays * depth.unsqueeze(-1)


def sliding_window_2d(
    x: torch.Tensor,
    window_size: Tuple[int, int],
    stride: int,
    *,
    dim: Tuple[int, int] = (-2, -1),
) -> torch.Tensor:
    """
    2D sliding window view using `Tensor.unfold`.

    Example:
      x: (H, W, C), dim=(0,1) -> (H', W', C, win_h, win_w)
      x: (..., H, W), dim=(-2,-1) -> (..., H', W', win_h, win_w)
    """
    if stride <= 0:
        raise ValueError(f"stride must be > 0, got {stride}")
    win_h, win_w = window_size
    if win_h <= 0 or win_w <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")

    dh, dw = dim
    dh = dh % x.dim()
    dw = dw % x.dim()
    if dh == dw:
        raise ValueError("dim must select two distinct dimensions")

    y = x.unfold(dh, win_h, stride)
    y = y.unfold(dw, win_w, stride)
    return y
