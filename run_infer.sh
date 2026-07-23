#!/bin/bash
device=6
name="p7"
input_dir="test/videos/${name}.mp4" 
out_dir="./output"

# Preprocess the reference video: estimating depth, camera poses, and segmenting the foreground
CUDA_VISIBLE_DEVICES=$device python inference.py \
    --image_dir ${input_dir} \
    --out_dir ${out_dir} \
    --mode 'dynamic_view_pre' \
    --stride 1 \
    --video_length 81 \
    --device 'cuda:0' \
    --height 480 \
    --width 832 \
    --radius_scale 1.0 \
    --traj_type 'freeze' \
    --d_phi -20 \
    --d_theta 0 \
    --x_offset 0 \
    --y_offset 0 \
    --z_offset 0 \
    --blip_path './checkpoints/blip2-opt-2.7b' \
    --transformer_path './checkpoints/UniView' \
    --model_name './checkpoints/Wan2.1-VACE-14B-diffusers' \
    --lora_path './checkpoints/loras/Wan21_CausVid_14B_T2V_lora_rank32.safetensors' \
    --stream3r_path './checkpoints/STream3R' \
    --diffusion_inference_steps 8 \
    --diffusion_guidance_scale 4.0 \
    --prompt '' \

# Preprocess the reference video: estimating optical flow and track
CUDA_VISIBLE_DEVICES=$device python mosca_precompute.py --cfg ./configs/recon_prep.yaml --ws ${out_dir}/${name}

# Align the static background to obtain more accurate camera poses and depth
CUDA_VISIBLE_DEVICES=$device python lite_moca_reconstruct.py --cfg ./configs/recon_fit.yaml --ws ${out_dir}/${name}

# Update depth
CUDA_VISIBLE_DEVICES=$device python inference.py \
    --image_dir ${input_dir} \
    --out_dir ${out_dir} \
    --mode 'align_depth' \
    --stride 1 \
    --video_length 81 \
    --device 'cuda:0' \
    --height 480 \
    --width 832 \
    --radius_scale 1.0 \
    --traj_type 'freeze' \
    --d_phi -20 \
    --d_theta 0 \
    --x_offset 0 \
    --y_offset 0 \
    --z_offset 0 \
    --blip_path './checkpoints/blip2-opt-2.7b' \
    --transformer_path './checkpoints/UniView' \
    --model_name './checkpoints/Wan2.1-VACE-14B-diffusers' \
    --lora_path './checkpoints/loras/Wan21_CausVid_14B_T2V_lora_rank32.safetensors' \
    --stream3r_path './checkpoints/STream3R' \
    --diffusion_inference_steps 8 \
    --diffusion_guidance_scale 4.0 \
    --prompt '' \

# Generate left supplementary views
CUDA_VISIBLE_DEVICES=$device python inference.py \
    --image_dir ${input_dir} \
    --out_dir ${out_dir} \
    --mode 'dynamic_view_left' \
    --stride 1 \
    --video_length 81 \
    --device 'cuda:0' \
    --height 480 \
    --width 832 \
    --radius_scale 1.0 \
    --traj_type 'freeze' \
    --d_phi -20 \
    --d_theta 0 \
    --x_offset 0 \
    --y_offset 0 \
    --z_offset 0 \
    --blip_path './checkpoints/blip2-opt-2.7b' \
    --transformer_path './checkpoints/UniView' \
    --model_name './checkpoints/Wan2.1-VACE-14B-diffusers' \
    --lora_path './checkpoints/loras/Wan21_CausVid_14B_T2V_lora_rank32.safetensors' \
    --stream3r_path './checkpoints/STream3R' \
    --diffusion_inference_steps 8 \
    --diffusion_guidance_scale 4.0 \
    --prompt '' \

# Generate right supplementary views
CUDA_VISIBLE_DEVICES=$device python inference.py \
    --image_dir ${input_dir} \
    --out_dir ${out_dir} \
    --mode 'dynamic_view_right' \
    --stride 1 \
    --video_length 81 \
    --device 'cuda:0' \
    --height 480 \
    --width 832 \
    --radius_scale 1.0 \
    --traj_type 'freeze' \
    --d_phi 20 \
    --d_theta 0 \
    --x_offset 0 \
    --y_offset 0 \
    --z_offset 0 \
    --blip_path './checkpoints/blip2-opt-2.7b' \
    --transformer_path './checkpoints/UniView' \
    --model_name './checkpoints/Wan2.1-VACE-14B-diffusers' \
    --lora_path './checkpoints/loras/Wan21_CausVid_14B_T2V_lora_rank32.safetensors' \
    --stream3r_path './checkpoints/STream3R' \
    --diffusion_inference_steps 8 \
    --diffusion_guidance_scale 4.0 \
    --prompt '' \

# 4D reconstruction
CUDA_VISIBLE_DEVICES=$device python mosca_reconstruct_input.py --cfg ./configs/recon_fit.yaml --ws ${out_dir}/${name}
