from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


CKPT_DIR = Path(__file__).resolve().parent


def _snapshot(repo_id: str, local_subdir: str) -> None:
    local_dir = CKPT_DIR / local_subdir
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )


def _download_file(repo_id: str, filename: str, local_subpath: str) -> None:
    dst = CKPT_DIR / local_subpath
    dst.parent.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(dst.parent),
        local_dir_use_symlinks=False,
    )


def main() -> None:
    # NOTE:
    # - Some orgs publish weights later; override repo IDs via env vars if needed.
    # - Large repos (diffusers / transformers) are downloaded via snapshot_download.
    # - Single-file weights are downloaded via hf_hub_download.

    uniview_repo = os.environ.get("UNIVIEW_TRANSFORMER_REPO", "Drexubery/UniView")
    wan_repo = os.environ.get("WAN_VACE_REPO", "Wan-AI/Wan2.1-VACE-14B-diffusers")
    blip_repo = os.environ.get("BLIP2_REPO", "Salesforce/blip2-opt-2.7b")
    stream3r_repo = os.environ.get("STREAM3R_REPO", "yslan/STream3R")

    moge_repo = os.environ.get("MOGE_REPO", "Ruicheng/moge-2-vitl-normal")
    sam2_repo = os.environ.get("SAM2_REPO", "facebook/sam2-hiera-large")
    tracer_repo = os.environ.get("TRACER_REPO", "Carve/tracer_b7")
    vda_repo = os.environ.get("VDA_REPO", "depth-anything/Video-Depth-Anything-Large")

    # Main model weights
    _snapshot(uniview_repo, "UniView")
    _snapshot(wan_repo, "Wan2.1-VACE-14B-diffusers")

    # Aux models
    _snapshot(blip_repo, "blip2-opt-2.7b")
    _snapshot(stream3r_repo, "STream3R")

    _download_file(moge_repo, "model.pt", "moge/model.pt")
    _download_file(sam2_repo, "sam2_hiera_large.pt", "sam2/sam2_hiera_large.pt")
    _download_file(tracer_repo, "tracer_b7.pth", "tracer_b7.pth")
    _download_file(vda_repo, "video_depth_anything_vitl.pth", "vda/video_depth_anything_vitl.pth")

    # Default LoRA download: CausVid v2 from Kijai/WanVideo_comfy.
    # Override env vars to switch versions, e.g.:
    #   WAN_LORA_FILENAME=Wan21_CausVid_14B_T2V_lora_rank32.safetensors python checkpoints/download_hf.py
    lora_repo = os.environ.get("WAN_LORA_REPO", "Kijai/WanVideo_comfy")
    lora_filename = os.environ.get("WAN_LORA_FILENAME", "Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors")
    if lora_repo:
        _download_file(lora_repo, lora_filename, f"loras/{lora_filename}")


if __name__ == "__main__":
    main()
