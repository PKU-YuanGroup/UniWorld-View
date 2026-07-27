#!/bin/bash
python app.py \
  --device "cuda:0" \
  --blip_path "./checkpoints/blip2-opt-2.7b" \
  --transformer_path "./checkpoints/UniView" \
  --model_name "./checkpoints/Wan2.1-VACE-14B-diffusers" \
  --lora_path "./checkpoints/loras/Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors" \
  --stream3r_path "./checkpoints/STream3R" \
  "$@"
