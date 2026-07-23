"""
Minimal `utils3d` shim.

The upstream MoGe project depends on a third-party `utils3d` package that pulls
in heavy optional dependencies (e.g. Open3D). UniView only needs a small subset
of tensor/array utilities used by the vendored MoGe code during inference.
"""

from . import torch  # noqa: F401
from . import numpy  # noqa: F401

