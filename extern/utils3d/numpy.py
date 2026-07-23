from __future__ import annotations

from typing import Tuple

import numpy as np


def image_uv(*, width: int, height: int, dtype=np.float32) -> np.ndarray:
    """
    Normalized UV in [0, 1] at pixel centers, shape (H, W, 2), (u, v).
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid size: width={width}, height={height}")
    xs = (np.arange(width, dtype=dtype) + 0.5) / float(width)
    ys = (np.arange(height, dtype=dtype) + 0.5) / float(height)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([grid_x, grid_y], axis=-1).astype(dtype, copy=False)


def image_pixel_center(*, width: int, height: int, dtype=np.float32) -> np.ndarray:
    uv01 = image_uv(width=width, height=height, dtype=dtype)
    scale = np.array([float(width), float(height)], dtype=dtype)
    return uv01 * scale


def sliding_window_2d(x: np.ndarray, window_size: Tuple[int, int], stride: int, *, axis: Tuple[int, int] = (-2, -1)) -> np.ndarray:
    """
    Numpy sliding window view (minimal; used by vendored MoGe utilities).
    """
    if stride != 1:
        raise NotImplementedError("Only stride=1 is supported in this minimal shim.")
    win_h, win_w = window_size
    if win_h <= 0 or win_w <= 0:
        raise ValueError(f"Invalid window_size: {window_size}")

    ax0, ax1 = axis
    ax0 = ax0 % x.ndim
    ax1 = ax1 % x.ndim
    if ax0 == ax1:
        raise ValueError("axis must select two distinct axes")

    # Move target axes to the end for simplicity
    perm = [i for i in range(x.ndim) if i not in (ax0, ax1)] + [ax0, ax1]
    x2 = np.transpose(x, perm)
    h, w = x2.shape[-2:]
    out_h = h - win_h + 1
    out_w = w - win_w + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError("window_size larger than input")

    shape = x2.shape[:-2] + (out_h, out_w, win_h, win_w)
    strides = x2.strides[:-2] + (x2.strides[-2], x2.strides[-1], x2.strides[-2], x2.strides[-1])
    view = np.lib.stride_tricks.as_strided(x2, shape=shape, strides=strides)
    return view

