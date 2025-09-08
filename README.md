# UniScene + VIPE Splat — Setup and Run Guide

This repo provides an interactive Gradio UI and CLI pipeline for dynamic novel view synthesis with an integrated NVIDIA VIPE (Video Pose Engine) pipeline.

Python 3.11 and a CUDA-enabled GPU are required. VIPE ships as source under `extern/vipe` and includes a prebuilt extension for CPython 3.11; if your Python differs or you switch PyTorch/CUDA, compile it locally as described below.

## 1) Environment Setup

- Python: 3.11
- GPU: NVIDIA CUDA 12.1 runtime compatible (per Torch 2.4.0+cu121)
- Recommended: Conda for Python and CUDA toolchain

Steps:

1. Create and activate a fresh env
   - `conda create -n vipe python=3.11 -y`
   - `conda activate vipe`

2. Install Python deps
   - `pip install -r requirements.txt`
   - Notes:
     - `requirements.txt` pins `torch==2.4.0+cu121` and sets the CUDA 12.1 wheel index.
     - Build tools `ninja`, `setuptools`, `wheel` are included for building VIPE.

3. (Optional but recommended for local builds) Install NVCC
   - You only need this if you will compile VIPE locally (see section 3). Either of:
     - `conda install -c nvidia cuda-compiler=12.1`
     - or `conda install -c conda-forge cuda-nvcc=12.1`

## 2) Quick VIPE Runtime Check (no compilation)

Use the bundled VIPE extension (prebuilt for CPython 3.11):

- `python vipe_test.py`
  - Prints shapes for depth/intrinsics/poses from `test/videos/1.mp4` using VIPE tensor stream.

If this works, the prebuilt `extern/vipe/vipe_ext*.so` matches your Python and you can proceed without compiling.

## 3) Compile VIPE Locally (when needed)

Compile VIPE if any of these apply:
- You changed Python version or Torch CUDA build
- The VIPE Runtime Check failed

Checklist:
- Ensure CUDA-enabled PyTorch is installed (`torch==2.4.0+cu121` from the requirements)
- Ensure NVCC is available (see section 1.3)
- Ensure build tools are present: `ninja`, `setuptools`, `wheel` (already in requirements)

Build commands:
- (Optional) set target architectures (For A100):
  - `export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"`
- Install in editable mode using your current env (no build isolation):
  - `pip install -e extern/vipe --no-build-isolation -v`

Verify:
- `python -c "import vipe, vipe_ext; print(vipe.__version__, vipe_ext.__file__)"`
- `python vipe_test.py`

Common issues:
- Torch is CPU-only or `torch.version.cuda is None` → reinstall the CUDA build of Torch per requirements
- `nvcc` not found → install NVCC via Conda and/or set `export PYTORCH_NVCC=$(which nvcc)`
- Unsupported SM arch → adjust `TORCH_CUDA_ARCH_LIST` per your GPU (A100: 8.0, H100: 9.0, etc.)

## 4) Running the Pipeline (CLI)

The CLI entry is `inference.py` and uses `configs/infer_config.py` to parse options. Minimal example using the provided sample video:

- `python inference.py \
    --image_dir test/videos/ori1.mp4 \
    --out_dir ./output/runs \
    --mode dynamic_view \
    --stride 1 \
    --video_length 81 \
    --device cuda:0 \
    --height 480 \
    --width 832`

Notes:
- For full diffusion rendering, you must provide model paths (BLIP, transformer, base model, LoRA). See section 6 for checkpoints.
- A convenience script `run_dynamic.sh` is provided but contains example absolute paths; update them to your local paths before running: `bash run_dynamic.sh`.

## 5) Running the Gradio UI

Start the UI:
- `python app_dynamic.py`

Notes:
- The UI code sets cache directories under `CACHE_ROOT` and overrides several model paths inside `app_dynamic.py`:
  - `opts.blip_path`
  - `opts.transformer_path`
  - `opts.model_name`
  - `opts.lora_path`
- Update these paths to your local checkpoints; otherwise, the UI will fail to load models.

## 6) 界面参数说明（Gradio）

以下说明与 UI 中的“静态视角（单图）”和“动态视角（视频）”面板对应。

静态视角（单图）
- 输入图片: 单张 RGB 图像。
- 相机俯仰角 `elevation`: 初始相机俯仰角（度），范围 [-45, 45]；数值越大表示越“俯视”。
- 相机移动半径 `center_scale`: 控制相机围绕场景运动的半径缩放，范围 [0.1, 2]；过大可能导致移动幅度过大。
- 相机轨迹 `pose`: 文本输入，格式为“经度φ; 纬度θ; x; y; z”的五段式参数序列，或输入 `swing` 使用内置摆动轨迹。
- 去噪步数 `steps`: 扩散模型推理步数，范围 [4, 10]，默认 8。该参数直接控制 `num_inference_steps`。
- 随机种子 `seed`: 控制可复现性。

动态视角（视频）
- 输入视频: mp4 等格式视频。
- 采帧间隔 `stride`: 从输入视频抽帧的间隔，范围 [1, 5]。
- 相机俯仰角 `elevation`: 初始相机俯仰角（度），范围 [-45, 45]；与单图一致，影响 `set_initial_camera(...)`。
- 相机移动半径 `center_scale`: 控制相机围绕场景运动的半径缩放，范围 [0.1, 2]。
- 相机轨迹 `pose`: 同上，支持“经度φ; 纬度θ; x; y; z”或 `swing`。
- 去噪步数 `steps`: 扩散模型推理步数，范围 [4, 10]，默认 8。UI 滑条的取值会传入并设置 `num_inference_steps`。
- 随机种子 `seed`: 控制可复现性。

渲染与输出
- 系统会保存若干中间可视化结果（例如 `input.mp4`, `render.mp4`, `mask.mp4`）以及最终的 `diffusion.mp4`。
- 动态视角下，先用 VIPE 估计每帧深度/内参/位姿，再统一到以首帧与 `elevation` 定义的世界坐标系进行渲染，最后送入扩散模型并做颜色校正。

提示
- 如果生成的视频移动幅度太大，可适当调小 `center_scale` 或调整 `pose`。
- 如需更快但可能略降质的结果，可适当降低 `steps`（例如 6），反之可提高至 10 以追求更稳的视觉效果（代价是更慢）。
