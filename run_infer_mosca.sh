#!/bin/bash

# MoSca-backed full pipeline:
#   1) Pre-process reference video (depth / optical flow / tracks).
#   2) Run MoCa static BA to align cameras and depth.
#   3) Run UniView inference with --geometry_backend mosca (reads the bundle).
#
# The MoSca precompute scripts (mosca_precompute.py, lite_moca_reconstruct.py)
# and their configs (configs/recon_prep.yaml, configs/recon_fit.yaml) already
# live in this repo. The bundle produced in step 2 is consumed by the mosca
# backend in demo.py via --mosca_ws.

# ----- edit these ------------------------------------------------------------

input_dir="./test/videos/2.mp4"
out_dir="./output"

# Auto-derive the experiment name from input_dir (e.g. test/videos/tiger.mp4 -> tiger).
# Override by exporting NAME before running this script if you want a custom name.
if [ -z "${NAME:-}" ]; then
  NAME="$(basename "${input_dir}")"
  NAME="${NAME%.*}"
fi
name="${NAME}"

# Video decode / sampling params passed to mosca_precompute.py.
video_length=81   # -1 = all frames
stride=1
height=480         # 0 = use max_res; explicit height when set
width=832          # 0 = use max_res; explicit width  when set
from_images=0      # 1 = skip decode, read PNGs from <ws>/images/
center_crop=1      # 1 = minimal-cover decode (matches demo.py)
# -----------------------------------------------------------------------------

ws=${out_dir}/${name}

# Step 1: extract frames + depth + optical flow + tracks.
echo "==> Step 1/3: mosca_precompute (ws=$ws)"
python mosca_precompute.py \
  --cfg ./configs/recon_prep.yaml \
  --ws "$ws" \
  --video_path "$input_dir" \
  --video_length "$video_length" \
  --stride "$stride" \
  $( [ "$height" -gt 0 ] && echo "--height $height" ) \
  $( [ "$width"  -gt 0 ] && echo "--width  $width" ) \
  $( [ "$from_images" -eq 1 ] && echo "--from_images" ) \
  $( [ "$center_crop" -eq 0 ] && echo "--no-center_crop" )

# Step 2: MoCa static BA -> ${ws}/bundle/bundle{,_cams}.pth.
echo "==> Step 2/3: lite_moca_reconstruct (ws=$ws)"
python lite_moca_reconstruct.py \
  --cfg ./configs/recon_fit.yaml \
  --ws "$ws"

# Step 3: UniView inference using the pre-computed bundle.
echo "==> Step 3/3: inference (geometry_backend=mosca, ws=$ws)"
python inference.py \
  --image_dir "$input_dir" \
  --out_dir "${out_dir}/${name}" \
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
  --ddim_steps 8 \
  --diffusion_guidance_scale 4.0 \
  --prompt '' \
  --geometry_backend 'mosca' \
  --mosca_ws "$ws" \
  "$@"