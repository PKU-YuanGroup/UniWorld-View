#!/bin/bash
python inference.py \
  --image_dir './test/videos/2.mp4' \
  --out_dir './output' \
  --mode 'dynamic_view' \
  --stride 1 \
  --video_length 81 \
  --device 'cuda:0' \
  --height 480 \
  --width 832 \
  --radius_scale 1.0 \
  --traj_type 'custom' \
  --d_phi 50 \
  --d_theta 0 \
  --x_offset 0 \
  --y_offset 0 \
  --z_offset 0 \
  --blip_path './checkpoints/blip2-opt-2.7b' \
  --transformer_path './checkpoints/UniView' \
  --model_name './checkpoints/Wan2.1-VACE-14B-diffusers' \
  --lora_path './checkpoints/loras/Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors' \
  --stream3r_path './checkpoints/STream3R' \
  --ddim_steps 8 \
  --diffusion_guidance_scale 4.0 \
  --prompt '' \
  "$@"
