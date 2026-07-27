#!/usr/bin/env bash
set -euo pipefail

# Downloads UniView / BLIP2 / STream3R / MoGe / SAM2 / Tracer / VDA / CausVid LoRA.
# Does NOT download Wan-AI/Wan2.1-VACE-14B-diffusers (provide it separately under
# checkpoints/Wan2.1-VACE-14B-diffusers).

# export WAN_LORA_REPO="${WAN_LORA_REPO:-Kijai/WanVideo_comfy}"
export WAN_LORA_FILENAME="${WAN_LORA_FILENAME:-Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors}"

python checkpoints/download_hf.py
