#!/bin/bash
# Ascend NPU-aware launcher for UniWorld-View inference.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

# --- Ascend / CANN runtime (required for torch_npu) ---
if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
if [[ -f /usr/local/Ascend/cann-9.0.0/set_env.sh ]]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/cann-9.0.0/set_env.sh
elif [[ -f /usr/local/Ascend/cann/set_env.sh ]]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/cann/set_env.sh
fi

# Prefer HF mirror in CN networks (optional override)
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# Skip NVIDIA CUDA JIT for vipe on Ascend
export VIPE_EXT_SKIP_JIT="${VIPE_EXT_SKIP_JIT:-1}"

# oom,optimization for memory reuse
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export MULTI_STREAM_MEMORY_REUSE=1
export COMBINED_ENABLE=1
export CPU_AFFINITY_CONF=1

# Default device: npu:0 when Ascend is present, else keep caller override via "$@"
DEFAULT_DEVICE="${UNIVIEW_DEVICE:-npu:0}"

# Save run logs under logs/ with a timestamp suffix
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/run_infer_${TIMESTAMP}.log"
echo "==> logging to: ${LOG_FILE}"

python inference.py \
  --image_dir './test/videos/2.mp4' \
  --out_dir './output' \
  --mode 'dynamic_view' \
  --stride 1 \
  --video_length 81 \
  --device "${DEFAULT_DEVICE}" \
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
  --diffusion_inference_steps 8 \
  --diffusion_guidance_scale 4.0 \
  --prompt '' \
  --no_align_with_vda \
  --render_method warp \
  "$@" \
  2>&1 | tee "${LOG_FILE}"
