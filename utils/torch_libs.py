from __future__ import annotations

import ctypes
import os
from typing import Optional


_LOADED = False
_LAST_ERROR: Optional[BaseException] = None


def ensure_torch_libs_loaded() -> None:
    """
    Ensure libtorch shared libraries are loaded into the current process.

    Some source-built extensions (e.g., PyTorch3D) may fail to import with:
      ImportError: libc10.so: cannot open shared object file

    because the dynamic loader cannot find torch's bundled libs under
    `<site-packages>/torch/lib` unless LD_LIBRARY_PATH is configured.

    Loading them explicitly via ctypes makes the import robust without
    requiring users to set environment variables.
    """
    global _LOADED, _LAST_ERROR
    if _LOADED:
        return

    try:
        import torch

        torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
        if not os.path.isdir(torch_lib_dir):
            _LOADED = True
            return

        names = [
            "libc10.so",
            "libtorch.so",
            "libtorch_cpu.so",
            "libc10_cuda.so",
            "libtorch_cuda.so",
            "libtorch_python.so",
        ]
        for name in names:
            path = os.path.join(torch_lib_dir, name)
            if os.path.exists(path):
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)

        _LOADED = True
    except BaseException as e:
        _LAST_ERROR = e
        raise

