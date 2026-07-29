from __future__ import annotations

import argparse
import os
import shutil
import tarfile
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


CKPT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Low-level download helpers
# ---------------------------------------------------------------------------

def _snapshot(repo_id: str, local_subdir: str) -> None:
    """Download an entire HF repo into ``CKPT_DIR / local_subdir``."""
    local_dir = CKPT_DIR / local_subdir
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )


def _download_file(repo_id: str, filename: str, local_subpath: str) -> None:
    """Download a single HF file into ``CKPT_DIR / local_subpath``."""
    dst = CKPT_DIR / local_subpath
    dst.parent.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(dst.parent),
        local_dir_use_symlinks=False,
    )


def _download_single_file_from_hf(repo_id: str, filename: str, dst: Path) -> None:
    """Download a single HF file directly into ``dst`` (no symlinks, no leftovers)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        print(f"[skip]   {dst} already exists ({dst.stat().st_size/1e6:.1f} MB)")
        return
    print(f"[file]   {repo_id}/{filename} -> {dst}")
    tmp = snapshot_download(
        repo_id=repo_id,
        allow_patterns=[filename],
        local_dir=str(dst.parent / ".tmp_hf_dl"),
        local_dir_use_symlinks=False,
    )
    src = Path(tmp) / filename
    if not src.exists():
        src = next(dst.parent.glob(f".tmp_hf_dl/**/{filename}"), None)
        if src is None:
            raise FileNotFoundError(
                f"Downloaded snapshot does not contain {filename}"
            )
    shutil.move(str(src), str(dst))
    shutil.rmtree(dst.parent / ".tmp_hf_dl", ignore_errors=True)


def _download_torch_hub_repo_from_github(
    github_user: str,
    github_repo: str,
    branch: str,
    dst: Path,
) -> None:
    """Download a GitHub repo tarball into ``dst`` (used for ``torch.hub`` cache)."""
    if (dst / "hubconf.py").exists():
        print(f"[skip]   {dst} already has hubconf.py")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://codeload.github.com/{github_user}/{github_repo}/tar.gz/refs/heads/{branch}"
    print(f"[github] {url} -> {dst}")
    tmp_tar = CKPT_DIR / ".tmp_repo.tar.gz"
    try:
        with urllib.request.urlopen(url, timeout=180) as resp, open(tmp_tar, "wb") as f:
            shutil.copyfileobj(resp, f)
        with tarfile.open(tmp_tar, "r:gz") as tar:
            # The top-level dir in the tarball is "<RepoName>-<branch>/".
            members = tar.getmembers()
            top_prefix = members[0].name.split("/")[0] if members else ""
            tar.extractall(path=dst.parent)
        extracted = dst.parent / top_prefix
        if extracted.exists() and extracted != dst:
            if dst.exists():
                shutil.rmtree(dst)
            extracted.rename(dst)
    finally:
        if tmp_tar.exists():
            tmp_tar.unlink()


# ---------------------------------------------------------------------------
# Optional module bundles
# ---------------------------------------------------------------------------
# Each function below pulls one optional bundle. Override the env vars to
# switch HF repos / filenames without editing this script.

def download_mosca() -> None:
    """MoSca auxiliary bundle: DepthCrafter + SVD + Metric3D."""
    print("==> MoSca auxiliary bundle")
    # Reserved for the marblueocean/UniView-mosca snapshot (not yet wired up).
    mosca_repo = os.environ.get("MOSCA_REPO", "Marblueocean/UniView-mosca")

    depthcrafter_repo = os.environ.get("DEPTHCRAFTER_REPO", "tencent/DepthCrafter")
    svd_repo = os.environ.get(
        "DEPTHCRAFTER_SVD_REPO", "stabilityai/stable-video-diffusion-img2vid-xt"
    )
    metric3d_repo = os.environ.get("METRIC3D_REPO", "JUGGHM/Metric3D")
    metric3d_filename = os.environ.get(
        "METRIC3D_FILENAME", "metric_depth_vit_giant2_800k.pth"
    )

    _snapshot(mosca_repo, "mosca")
    _snapshot(depthcrafter_repo, "depthcrafter")
    _snapshot(svd_repo, "stable-video-diffusion-img2vid-xt")
    _download_single_file_from_hf(
        repo_id=metric3d_repo,
        filename=metric3d_filename,
        dst=CKPT_DIR / "metric3d" / metric3d_filename,
    )
    _download_torch_hub_repo_from_github(
        github_user="Yvanyin",
        github_repo="metric3d",
        branch="main",
        dst=CKPT_DIR / "torch_hub_cache" / "Yvanyin_metric3d",
    )


def download_stream3r() -> None:
    """STream3R auxiliary bundle (alternative to MoSca)."""
    print("==> STream3R auxiliary bundle")
    stream3r_repo = os.environ.get("STREAM3R_REPO", "yslan/STream3R")
    _snapshot(stream3r_repo, "STream3R")


def download_recon() -> None:
    """Reconstruction bundle: ROSE + Wan2.1-Fun-1.3B-InP."""
    print("==> Reconstruction bundle")
    rose_repo = os.environ.get("ROSE_REPO", "Kunbyte/ROSE")
    wan_inp_repo = os.environ.get("WAN_INP_REPO", "alibaba-pai/Wan2.1-Fun-1.3B-InP")
    _snapshot(rose_repo, "rose/transformer")
    _snapshot(wan_inp_repo, "rose/Wan2.1-Fun-1.3B-InP")


# ---------------------------------------------------------------------------
# Core pipeline weights (always downloaded)
# ---------------------------------------------------------------------------

def download_core() -> None:
    """Core UniView pipeline weights (transformer, Wan2.1-VACE, BLIP2, MoGe, SAM2, tracer)."""
    print("==> Core UniView pipeline")
    uniview_repo = os.environ.get("UNIVIEW_TRANSFORMER_REPO", "Drexubery/UniView")
    wan_repo = os.environ.get("WAN_VACE_REPO", "Wan-AI/Wan2.1-VACE-14B-diffusers")
    blip_repo = os.environ.get("BLIP2_REPO", "Salesforce/blip2-opt-2.7b")
    moge_repo = os.environ.get("MOGE_REPO", "Ruicheng/moge-2-vitl-normal")
    sam2_repo = os.environ.get("SAM2_REPO", "facebook/sam2-hiera-large")
    tracer_repo = os.environ.get("TRACER_REPO", "Carve/tracer_b7")

    _snapshot(uniview_repo, "UniView")
    _snapshot(wan_repo, "Wan2.1-VACE-14B-diffusers")
    _snapshot(blip_repo, "blip2-opt-2.7b")
    _download_file(moge_repo, "model.pt", "moge/model.pt")
    _download_file(sam2_repo, "sam2_hiera_large.pt", "sam2/sam2_hiera_large.pt")
    _download_file(tracer_repo, "tracer_b7.pth", "tracer_b7.pth")


def download_lora() -> None:
    """CausVid LoRA (always downloaded when ``WAN_LORA_REPO`` is set)."""
    lora_repo = os.environ.get("WAN_LORA_REPO", "Kijai/WanVideo_comfy")
    lora_filename = os.environ.get(
        "WAN_LORA_FILENAME", "Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors"
    )
    if not lora_repo:
        return
    print("==> CausVid LoRA")
    _download_file(lora_repo, lora_filename, f"loras/{lora_filename}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the optional module bundles."""
    parser = argparse.ArgumentParser(
        description=(
            "Download UniView checkpoints from Hugging Face. The core pipeline "
            "weights and the CausVid LoRA are always pulled; use the flags "
            "below to control whether each optional bundle is downloaded."
        ),
    )
    parser.add_argument(
        "--mosca",
        action="store_true",
        default=False,
        help="Download the MoSca auxiliary bundle (DepthCrafter, SVD, Metric3D).",
    )
    parser.add_argument(
        "--stream3r",
        action="store_true",
        default=False,
        help="Download the STream3R auxiliary bundle (alternative to --mosca).",
    )
    parser.add_argument(
        "--recon",
        action="store_true",
        default=False,
        help="Download the reconstruction bundle (ROSE + Wan2.1-Fun-1.3B-InP).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Required weights are always downloaded.
    download_core()
    download_lora()

    # Optional bundles — only pulled when the corresponding flag is passed.
    selected = []
    if args.mosca:
        download_mosca()
        selected.append("mosca")
    if args.stream3r:
        download_stream3r()
        selected.append("stream3r")
    if args.recon:
        download_recon()
        selected.append("recon")

    if not selected:
        print(
            "[info] No optional bundles selected. "
            "Pass --mosca / --stream3r / --recon to pull extras."
        )
    else:
        print(f"[done] Optional bundles downloaded: {', '.join(selected)}")


if __name__ == "__main__":
    main()