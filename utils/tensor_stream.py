# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Iterable, List

import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
try:
    from vipe.streams.base import VideoFrame, VideoStream  # optional dependency
    _VIPE_AVAILABLE = True
except Exception:
    VideoFrame = None  # type: ignore
    VideoStream = None  # type: ignore
    _VIPE_AVAILABLE = False


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


if _VIPE_AVAILABLE:
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


def _preprocess_frames_for_stream3r(frames_np: np.ndarray, mode: str = "crop") -> torch.Tensor:
    """
    Mirror STream3R's load_and_preprocess_images but for in-memory video frames.
    - Width resized to 518 (divisible by 14). Height resized proportionally and rounded to /14.
    - For crop mode: if height > 518 after resize, center crop to 518.
    - If shapes still differ across frames, pad to max(H, W) with white (1.0).

    Args:
        frames_np: np.ndarray of shape (T, H, W, 3), values in [0,1]
        mode: 'crop' or 'pad' (keep 'crop' to match default)

    Returns:
        Tensor of shape (T, 3, Hs, Ws) in [0,1]
    """
    assert frames_np.ndim == 4 and frames_np.shape[-1] == 3
    T = frames_np.shape[0]
    target_size = 518
    images: List[torch.Tensor] = []
    shapes = set()

    to_pil = ToPILImage()
    to_tensor = ToTensor()

    for i in range(T):
        # Convert to PIL in [0,255]
        img_np = np.clip(frames_np[i] * 255.0, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_np).convert("RGB")
        width, height = img.size

        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14
        else:  # crop mode
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14

        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img_t = to_tensor(img)  # (3,H,W) in [0,1]

        # Center-crop height if necessary (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img_t = img_t[:, start_y : start_y + target_size, :]

        if mode == "pad":
            h_padding = target_size - img_t.shape[1]
            w_padding = target_size - img_t.shape[2]
            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                img_t = F.pad(img_t, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0)

        shapes.add((img_t.shape[1], img_t.shape[2]))
        images.append(img_t)

    # If different shapes, pad to the max H and W (white background)
    if len(shapes) > 1:
        max_height = max(s[0] for s in shapes)
        max_width = max(s[1] for s in shapes)
        padded_images: List[torch.Tensor] = []
        for img_t in images:
            h_padding = max_height - img_t.shape[1]
            w_padding = max_width - img_t.shape[2]
            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                img_t = F.pad(img_t, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0)
            padded_images.append(img_t)
        images = padded_images

    return torch.stack(images, dim=0)
