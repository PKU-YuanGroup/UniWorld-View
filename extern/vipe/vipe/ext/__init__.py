# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""vipe CUDA extensions loader.

On Ascend NPU / CPU (no NVIDIA CUDA toolkit), skip JIT compile so pure-Python
submodules such as ``vipe.ext.xformers`` remain importable.
"""

from __future__ import annotations

import os
import warnings

_C = None
vipe_ext_not_found = True

try:
    import vipe_ext as _C  # type: ignore

    vipe_ext_not_found = False
except ImportError:
    vipe_ext_not_found = True


def _cuda_toolkit_available() -> bool:
    try:
        import torch

        if not torch.cuda.is_available():
            return False
    except Exception:
        return False
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home and os.path.isdir(cuda_home):
        return True
    # nvcc on PATH is also acceptable
    from shutil import which

    return which("nvcc") is not None


_want_jit = vipe_ext_not_found or os.environ.get("VIPE_EXT_JIT", "0") == "1"
_force_skip = os.environ.get("VIPE_EXT_SKIP_JIT", "0") == "1"

if _want_jit and not _force_skip and _cuda_toolkit_available():
    from torch.utils.cpp_extension import load

    from vipe.ext.specs import get_cpp_flags, get_cuda_flags, get_sources

    _C = load(
        name="vipe_ext_jit",
        sources=get_sources(),
        extra_cflags=get_cpp_flags(),
        extra_cuda_cflags=get_cuda_flags(),
        verbose=True,
    )
    vipe_ext_not_found = False
elif _want_jit:
    warnings.warn(
        "Skipping vipe CUDA extension JIT build "
        "(no CUDA toolkit / CUDA_HOME, or VIPE_EXT_SKIP_JIT=1). "
        "Pure-Python fallbacks (e.g. vipe.ext.xformers) remain available; "
        "CUDA-only SLAM/corr ops will be unavailable."
    )


class _MissingExt:
    def __getattr__(self, name: str):
        raise ImportError(
            f"vipe extension '{name}' is unavailable because CUDA extensions "
            "were not built. Install NVIDIA CUDA toolkit or disable features "
            "that require vipe CUDA ops."
        )


if _C is None:
    _C = _MissingExt()

# Reference to submodules (may raise on access if extensions missing)
droid_net_ext = getattr(_C, "droid_net_ext", None)
grounding_dino_ext = getattr(_C, "grounding_dino_ext", None)
utils_ext = getattr(_C, "utils_ext", None)
slam_ext = getattr(_C, "slam_ext", None)
scatter_ext = getattr(_C, "scatter_ext", None)
lietorch_ext = getattr(_C, "lietorch_ext", None)
corr_ext = getattr(_C, "corr_ext", None)
