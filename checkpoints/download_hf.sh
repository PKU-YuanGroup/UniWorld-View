#!/usr/bin/env bash
set -euo pipefail

export WAN_LORA_REPO="${WAN_LORA_REPO:-Kijai/WanVideo_comfy}"
export WAN_LORA_FILENAME="${WAN_LORA_FILENAME:-Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors}"

python checkpoints/download_hf.py
