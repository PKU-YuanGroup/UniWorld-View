#!/usr/bin/env bash
set -euo pipefail

export WAN_LORA_REPO="${WAN_LORA_REPO:-Kijai/WanVideo_comfy}"
export WAN_LORA_FILENAME="${WAN_LORA_FILENAME:-Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors}"

# Core weights are always downloaded. The three optional bundles
# (--mosca, --stream3r, --recon) are off by default; pass the flag
# explicitly to pull each one, e.g.:
#   bash checkpoints/download_hf.sh --mosca --recon
python checkpoints/download_hf.py "$@"