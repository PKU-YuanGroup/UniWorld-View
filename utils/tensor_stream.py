# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Iterable

import torch

from vipe.streams.base import VideoFrame, VideoStream


def _to_rgb_float01(frame: torch.Tensor) -> torch.Tensor:
    """
    Normalize an input frame to shape (H, W, 3), float32 in [0, 1].
    Accepts (H,W,3) or (3,H,W); uint8/float.
    """
    if frame.dim() != 3:
        raise ValueError(f"Expected 3D frame tensor, got shape {tuple(frame.shape)}")

    # Channel last (H, W, 3)
    if frame.shape[-1] == 3:
        rgb = frame
    elif frame.shape[0] == 3:
        rgb = frame.permute(1, 2, 0)
    else:
        raise ValueError(
            f"Unsupported frame shape {tuple(frame.shape)}; expected (H,W,3) or (3,H,W)."
        )

    # Convert dtypes and range
    if rgb.dtype == torch.uint8:
        rgb = rgb.to(dtype=torch.float32) / 255.0
    else:
        rgb = rgb.to(dtype=torch.float32)
        # Assume already 0-1 if not uint8.

    return rgb


class TensorVideoStream(VideoStream):
    """
    A video stream backed by a preloaded tensor or iterable of frame tensors.

    - Accepts `video` as:
      - torch.Tensor with shape (T, H, W, 3) or (T, 3, H, W), or
      - Iterable[torch.Tensor], each (H, W, 3) or (3, H, W).
    - Frames are converted to float32 [0,1] and moved to CUDA by default.
    """

    def __init__(
        self,
        video: torch.Tensor | Iterable[torch.Tensor],
        *,
        fps: float = 30.0,
        name: str = "tensor",
        device: torch.device | str = "cuda",
    ) -> None:
        super().__init__()
        self._name = name
        self._fps = float(fps)

        # Normalize to list of frames on target device
        if isinstance(video, torch.Tensor):
            if video.dim() != 4:
                raise ValueError(
                    f"TensorVideoStream expects 4D tensor (T,H,W,3) or (T,3,H,W); got {tuple(video.shape)}"
                )
            if video.shape[-1] == 3 or video.shape[1] == 3:
                frames = [video[i] for i in range(video.shape[0])]
            else:
                raise ValueError(
                    f"Last or second dim must be 3 (RGB channels); got {tuple(video.shape)}"
                )
        else:
            frames = list(video)
            if len(frames) == 0:
                raise ValueError("Empty video iterable provided to TensorVideoStream")

        self._frames = []
        for f in frames:
            rgb = _to_rgb_float01(f)
            self._frames.append(rgb.to(device, non_blocking=True).contiguous())

        h, w = self._frames[0].shape[:2]
        self._height = int(h)
        self._width = int(w)

    def frame_size(self) -> tuple[int, int]:
        return (self._height, self._width)

    def fps(self) -> float:
        return self._fps

    def name(self) -> str:
        return self._name

    def __len__(self) -> int:
        return len(self._frames)

    def __iter__(self):
        for idx, rgb in enumerate(self._frames):
            # rgb already on target device and in [0,1]
            yield VideoFrame(raw_frame_idx=idx, rgb=rgb)
