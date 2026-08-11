## ___***UniWorld-View: Large-Baseline View Synthesis via Video Diffusion Models***___
<div align="center">

 <a href='https://github.com/PKU-YuanGroup/UniWorld-View'><img src='https://img.shields.io/badge/GitHub-UniWorld-View-blue.svg'></a> &nbsp;
 <a href='https://huggingface.co/Drexubery/UniView'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a> &nbsp;
 <a href='https://www.hiascend.com/'><img src='https://img.shields.io/badge/Ascend-NPU-red.svg'></a> &nbsp;

</div>

🤗 If you find UniWorld-View useful, **please help ⭐ this repo**, which is important to Open-Source projects. Thanks!

> **This tree (`npu` branch / UniWorld-View) is the Ascend NPU inference port.**  
> For the original NVIDIA GPU workflow, see the upstream `master` branch or `UniWorld-View_original`.

## 🔆 Introduction

UniWorld-View can generate high-fidelity novel views from <strong>casually captured monocular video</strong> or <strong>single images</strong>, while also supporting highly precise pose control.

<!-- Add demo GIFs or images here when available, e.g.:
<table class="center">
    <tr style="font-weight: bolder;">
        <td>Input Video / Image &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Novel View</td>
    </tr>
  <tr>
    <td>
    <img src=assets/demo.gif style="width: 100%; height: auto;">
  </td>
  </tr>
</table>
-->

For 4D reconstruction, please switch to the `recon` branch (GPU-oriented; not covered by this NPU port).

## ⚙️ Setup

### 0. Hardware & software requirement

| Item | Recommendation |
|------|----------------|
| Device | Ascend NPU (e.g. Atlas), **HBM ≥ 60GB** preferred |
| Driver / Toolkit | Ascend Driver + `ascend-toolkit` + CANN (this machine uses paths under `/usr/local/Ascend/`, e.g. `cann-9.0.0`) |
| Python | **3.10** (tested) |
| Framework | PyTorch + **torch_npu** matching your CANN version |

Wan-14B and auxiliary models cannot all stay on one ~61GB NPU at once. On NPU the diffusion pipeline is kept on CPU by default and moved to device when needed (`utils/device_compat.py` / `demo.py`).

Ensure no other large jobs (e.g. vLLM) occupy NPU HBM before inference (`npu-smi`).

### 1. Clone this repo

```bash
git clone -b npu https://github.com/PKU-YuanGroup/UniWorld-View.git
cd UniWorld-View
```

Inference expects the following third-party repos under `extern/`:

```bash
git clone https://github.com/NIRVANALAN/STream3R.git extern/STream3R
git clone https://github.com/nv-tlabs/vipe.git extern/vipe
```

Expected layout:

```text
UniWorld-View/
  extern/
    STream3R/
    vipe/
    vipe_patches/   # optional GPU patches; see below
```

**Ascend note:** `vipe`’s NVIDIA CUDA extension must not be JIT-compiled on NPU. This port:

- sets `VIPE_EXT_SKIP_JIT=1` in `run_infer.sh`
- patches `extern/vipe/vipe/ext/__init__.py` to skip CUDA JIT when CUDA is unavailable
- **defaults to `--no_align_with_vda`** (Video-Depth-Anything / vipe CUDA path is not enabled on NPU)

If you later enable `--align_with_vda` on a CUDA machine, apply the GPU device patches as in the original README:

```bash
cp extern/vipe_patches/videodepthanything/__init__.py \
  extern/vipe/vipe/priors/depth/videodepthanything/__init__.py
cp extern/vipe_patches/videodepthanything/video_depth.py \
  extern/vipe/vipe/priors/depth/videodepthanything/video_depth.py
```

### 2. Setup environment

#### 2.1 Source Ascend / CANN (required)

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# Prefer the versioned CANN install if present:
source /usr/local/Ascend/cann-9.0.0/set_env.sh
# or: source /usr/local/Ascend/cann/set_env.sh
```

`run_infer.sh` sources these automatically when the files exist. Without them, `torch.npu` stays unavailable even if `/dev/davinci*` is present.

#### 2.2 Python packages

```bash
conda create -n uniworld-view python=3.10 -y
conda activate uniworld-view

# Install PyTorch + torch_npu for your Ascend/CANN version
# (use the wheel set from Huawei Ascend docs that matches your CANN)
# Example shape only — replace with the official torch / torch_npu pair for your toolkit:
#   pip install torch==... torch_npu==... torchvision==...

pip install -r requirements.txt
pip install carvekit --no-deps
# Headless servers (no libGL): prefer OpenCV without GUI deps
pip install opencv-python-headless==4.11.0.86
conda install -y -c conda-forge eigen
```

Optional HF mirror (already defaulted in `run_infer.sh`):

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

#### 2.3 Render backend (`--render_method`)

| Method | Needs PyTorch3D | Ascend default |
|--------|-----------------|----------------|
| `warp` | No | **Yes** (`run_infer.sh`) |
| `hybrid` / `mesh` | Yes (CUDA `nvcc` build) | Optional; usually keep `warp` on NPU |

Official `extern/install_pytorch3d.sh` targets CUDA. On Ascend, prefer:

```bash
bash run_infer.sh   # already passes --render_method warp
```

### 3. Download pretrained models

By default, UniWorld-View loads weights from `./checkpoints/` (see `checkpoints/README.md`).

#### Option A (recommended): download to local `./checkpoints/`

```bash
export HF_ENDPOINT=https://hf-mirror.com
bash checkpoints/download_hf.sh
```

This downloads: UniWorld-View transformer, BLIP2, STream3R, MoGe, SAM2, TracerB7, Video-Depth-Anything, and the default CausVid LoRA `v2`.

**Wan2.1-VACE-14B-diffusers is not downloaded by the script** — place it under:

```text
checkpoints/Wan2.1-VACE-14B-diffusers/
```

#### Option B: load from Hugging Face Hub (cache)

Run inference with `--no_load_weights_locally` and override repo IDs via environment variables or CLI (see `checkpoints/download_hf.py`).

## 💫 Inference

Run from the **repo root**.

### 1. Command line (Ascend)

```bash
bash run_infer.sh
```

What the launcher does:

- sources Ascend / CANN env
- sets `HF_ENDPOINT`, `VIPE_EXT_SKIP_JIT=1`, and NPU memory env vars
- defaults `--device npu:0`, `--no_align_with_vda`, `--render_method warp`
- tees logs to `logs/run_infer_YYYYMMDD_HHMMSS.log`

Use another NPU:

```bash
UNIVIEW_DEVICE=npu:3 bash run_infer.sh
# or
bash run_infer.sh --device npu:3
```

Extra CLI args are forwarded after the defaults, e.g.:

```bash
bash run_infer.sh --image_dir './test/videos/2.mp4' --out_dir './output'
```

Device helpers live in `utils/device_compat.py` (`npu:N` / `cuda:N` / `cpu`). On Ascend-only hosts, a leftover `cuda:0` default is remapped to `npu:*` in `inference.py`.

### 2. Local Gradio demo

```bash
# Source Ascend env first (same as §2.1), then:
bash run_app.sh --device npu:0
```

`run_app.sh` still ships a CUDA-style default; **always pass `--device npu:N` on Ascend**.

Open `http://127.0.0.1:7860`. For remote access:

```bash
GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT=7860 bash run_app.sh --device npu:0
```

## 🔧 Ascend-specific notes

| Topic | Behavior on this port |
|-------|------------------------|
| Default device | `npu:0` (`configs/infer_config.py`, `run_infer.sh`) |
| VDA depth align | Off by default (`--no_align_with_vda`); lazy-import when enabled |
| Memory | Pipeline CPU-offload style on NPU; set `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` (in launcher) |
| Troubleshooting | See [`ASCEND_NPU_ISSUES.md`](./ASCEND_NPU_ISSUES.md) |

Quick self-check after env setup:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann-9.0.0/set_env.sh
python -c "import torch, torch_npu; print(torch.npu.is_available(), torch.npu.device_count())"
npu-smi info
```

## 📢 Limitations

Our model excels at handling videos with well-defined objects and clear motion, as demonstrated in the demo videos. However, since it is built upon a pretrained video diffusion model, it may struggle with complex cases that go beyond the generation capabilities of the base model.

On Ascend NPU specifically:

- `--align_with_vda` / `vipe` CUDA extensions are not supported in the default path
- `hybrid` / `mesh` rendering depends on PyTorch3D (CUDA-oriented); use `warp` unless you have a working NPU/CPU build
- Free enough NPU HBM; concurrent large jobs can cause OOM

## 🤗 Related Works

Including but not limited to: [VACE](https://github.com/ali-vilab/VACE), [ViewCrafter](https://github.com/Drexubery/ViewCrafter), [GCD](https://gcd.cs.columbia.edu/), [NVS-Solver](https://github.com/ZHU-Zhiyu/NVS_Solver), [DimensionX](https://github.com/wenqsun/DimensionX), [ReCapture](https://generative-video-camera-controls.github.io/), [TrajAttention](https://xizaoqu.github.io/trajattn/), [GS-DiT](https://wkbian.github.io/Projects/GS-DiT/), [DaS](https://igl-hkust.github.io/das/), [RecamMaster](https://github.com/KwaiVGI/ReCamMaster), [GEN3C](https://research.nvidia.com/labs/toronto-ai/GEN3C/), [CAT4D](https://cat-4d.github.io/)...

## 📜 Citation

If you find this work helpful, please consider citing:

```BibTeX
@misc{UniWorld-View2025,
    author    = {PKU-YuanGroup},
    title     = {UniWorld-View: Large-Baseline View Synthesis via Video Diffusion Models},
    year      = {2025},
    url       = {https://github.com/PKU-YuanGroup/UniWorld-View}
}
```

## License

This project is licensed under Apache-2.0. See `LICENSE`.
