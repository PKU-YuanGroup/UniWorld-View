"""Device helpers for NVIDIA CUDA and Ascend NPU."""
from __future__ import annotations

import os
import warnings
from contextlib import contextmanager, nullcontext
from typing import Iterator, Optional

import torch

_NPU_READY = False


def setup_backend() -> None:
    """Import torch_npu when Ascend runtime is present."""
    global _NPU_READY
    if _NPU_READY:
        return
    try:
        import torch_npu  # noqa: F401

        _NPU_READY = True
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"torch_npu not loaded: {exc}")


def resolve_device(device: Optional[str] = None) -> str:
    """Pick a usable device string. Prefer explicit arg, else NPU, else CUDA, else CPU."""
    setup_backend()
    if device:
        return str(device)

    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu:0"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def device_type(device: str | torch.device) -> str:
    if isinstance(device, torch.device):
        return device.type
    s = str(device)
    if ":" in s:
        return s.split(":", 1)[0]
    return s


def is_accelerator_available() -> bool:
    setup_backend()
    if hasattr(torch, "npu") and torch.npu.is_available():
        return True
    return torch.cuda.is_available()


def empty_cache() -> None:
    setup_backend()
    if hasattr(torch, "npu") and torch.npu.is_available():
        try:
            torch.npu.empty_cache()
        except Exception:
            pass
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def autocast_device_type(device: str | torch.device | None = None) -> str:
    """Return amp device type string for torch.autocast / torch.amp.autocast."""
    if device is None:
        if hasattr(torch, "npu") and torch.npu.is_available():
            return "npu"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return device_type(device)


@contextmanager
def amp_autocast(device: str | torch.device | None = None, **kwargs) -> Iterator[None]:
    dtype = kwargs.pop("dtype", None)
    enabled = kwargs.pop("enabled", True)
    dev_t = autocast_device_type(device)
    if not enabled or dev_t == "cpu":
        yield
        return
    # Prefer torch.amp.autocast; fall back to torch.autocast.
    try:
        cm = torch.amp.autocast(dev_t, dtype=dtype, enabled=True) if dtype is not None else torch.amp.autocast(
            dev_t, enabled=True
        )
    except Exception:
        try:
            cm = torch.autocast(device_type=dev_t, dtype=dtype, enabled=True) if dtype is not None else torch.autocast(
                device_type=dev_t, enabled=True
            )
        except Exception:
            cm = nullcontext()
    with cm:
        yield


def make_generator(device: str | torch.device, seed: int = 42) -> torch.Generator:
    """Create a RNG. Some backends only support CPU generators."""
    dev_t = device_type(device)
    try:
        gen = torch.Generator(device=str(device) if dev_t != "npu" else "cpu")
    except Exception:
        gen = torch.Generator(device="cpu")
    return gen.manual_seed(int(seed))


def ensure_ascend_hint() -> None:
    """Warn if NPU devices exist in OS but torch cannot see them (env not sourced)."""
    if os.path.exists("/dev/davinci0") and not (
        hasattr(torch, "npu") and torch.npu.is_available()
    ):
        warnings.warn(
            "Ascend device node /dev/davinci0 exists but torch.npu is unavailable. "
            "Source Ascend/CANN env before running, e.g. "
            "`source /usr/local/Ascend/ascend-toolkit/set_env.sh` and "
            "`source /usr/local/Ascend/cann-9.0.0/set_env.sh`."
        )
