#!/usr/bin/env bash
set -euo pipefail

# This script assumes:
# - You are already inside the target python/conda environment.
# - You have a working CUDA toolchain (nvcc) and a C++ compiler.
#
# It installs PyTorch3D from source at a pinned commit tested with:
#   Python 3.10/3.11 + torch==2.4.0+cu121

PYTORCH3D_COMMIT="${PYTORCH3D_COMMIT:-71db7a0ea293f2626f5ea1c70366870a607129d0}"

if ! command -v nvcc >/dev/null 2>&1; then
  echo "ERROR: nvcc not found. Please install CUDA toolkit and ensure nvcc is in PATH." >&2
  exit 1
fi

if ! command -v g++ >/dev/null 2>&1 && ! command -v c++ >/dev/null 2>&1; then
  echo "ERROR: a C++ compiler (g++/c++) is required to build PyTorch3D." >&2
  exit 1
fi

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

python -m pip install -U pip setuptools wheel
python -m pip install -U ninja fvcore iopath

WORKDIR="${WORKDIR:-/tmp/pytorch3d}"
rm -rf "${WORKDIR}"
export WORKDIR
export PYTORCH3D_COMMIT

# Prefer a local source if provided (no network required).
if [ -n "${PYTORCH3D_SOURCE:-}" ] && [ -d "${PYTORCH3D_SOURCE}" ]; then
  cp -a "${PYTORCH3D_SOURCE}" "${WORKDIR}"
else
  # Download a source tarball (more robust than `git clone` on some networks).
  python - <<PY
import io
import os
import tarfile
import urllib.request
from pathlib import Path

commit = os.environ["PYTORCH3D_COMMIT"]
workdir = Path(os.environ["WORKDIR"]).resolve()
url = f"https://codeload.github.com/facebookresearch/pytorch3d/tar.gz/{commit}"

print("Downloading:", url)
with urllib.request.urlopen(url) as r:
    data = r.read()

tmp = workdir.parent / f"pytorch3d-{commit}"
if tmp.exists():
    import shutil
    shutil.rmtree(tmp)

with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
    tf.extractall(path=workdir.parent)

extracted = workdir.parent / f"pytorch3d-{commit}"
if not extracted.exists():
    # Some tarballs extract to pytorch3d-<sha>, not commit id; fall back to first dir.
    dirs = [p for p in workdir.parent.iterdir() if p.is_dir() and p.name.startswith("pytorch3d-")]
    if not dirs:
        raise RuntimeError("Failed to find extracted pytorch3d source directory.")
    extracted = sorted(dirs)[-1]

extracted.rename(workdir)
print("Source ready:", workdir)
PY
fi

export FORCE_CUDA=1
export MAX_JOBS="${MAX_JOBS:-8}"
export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"

python -m pip install -e "${WORKDIR}" --no-build-isolation

python - <<'PY'
import ctypes
import os

import torch

torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
for name in (
    "libc10.so",
    "libtorch.so",
    "libtorch_cpu.so",
    "libc10_cuda.so",
    "libtorch_cuda.so",
    "libtorch_python.so",
):
    path = os.path.join(torch_lib_dir, name)
    if os.path.exists(path):
        ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)

import pytorch3d, pytorch3d._C

print("pytorch3d ok")
PY
