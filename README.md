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
