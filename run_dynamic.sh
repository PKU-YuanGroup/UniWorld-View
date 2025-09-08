#!/usr/bin/env bash

python inference.py \
  --image_dir 'test/videos/ori1.mp4' \
  --out_dir './output/gradio/test_run' \
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
  --blip_path '/mnt/workspace/ywb/cogvideo_ViewCrafter/checkpoints/blip2-opt-2.7b' \
  --transformer_path '/mnt/workspace/ywb/VideoX-Fun/experiments/good_04gt_block0_openvid_dl3dv_real10k_4-1-1/checkpoint-50_good/transformer' \
  --model_name '/mnt/workspace/ywb/VideoX-Fun/Wan2.1-VACE-14B-diffusers' \
  --lora_path '/mnt/workspace/ywb/VideoX-Fun/loras/Wan21_CausVid_14B_T2V_lora_rank32.safetensors' \
  --diffusion_inference_steps 4 \
  --diffusion_guidance_scale 1.0 \
  --prompt '' \
