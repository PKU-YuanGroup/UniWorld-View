## ___***UniWorld-View: Large-Baseline View Synthesis via Video Diffusion Models***___
<div align="center">

 <a href='https://github.com/PKU-YuanGroup/UniWorld-View'><img src='https://img.shields.io/badge/GitHub-UniWorld-View-blue.svg'></a> &nbsp;
 <a href='https://huggingface.co/Drexubery/UniView'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a> &nbsp;

</div>

🤗 If you find UniWorld-View useful, **please help ⭐ this repo**, which is important to Open-Source projects. Thanks!

## 🔥🔥🔥 Latest News
- 🎉 Jul, 2026: UniWorld-View ranked **1st** on the [WorldScore](https://huggingface.co/spaces/Howieeeee/WorldScore_Leaderboard) Leaderboard (by Stanford Prof. Fei-Fei Li's Team)

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

For 4D reconstruction, please switch to the `recon` branch.

## ⚙️ Setup

### 0. GPU memory requirement

We recommend a GPU with **VRAM ≥ 60GB** for smooth inference.

### 1. Clone this repo

```bash
git clone https://github.com/PKU-YuanGroup/UniWorld-View.git
cd UniWorld-View
```

Inference expects the following third-party repos to exist under `extern/`:

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
```

To make `--device cuda:N` work correctly for `vipe` / VDA on non-default GPUs, replace two files in the official `vipe` repo with the patched copies shipped in `extern/vipe_patches/`:

```bash
cp extern/vipe_patches/videodepthanything/__init__.py \
  extern/vipe/vipe/priors/depth/videodepthanything/__init__.py
cp extern/vipe_patches/videodepthanything/video_depth.py \
  extern/vipe/vipe/priors/depth/videodepthanything/video_depth.py
```

### 2. Setup environment

```bash
conda create -n uniworld-view python=3.10 -y
conda activate uniworld-view
pip install torch==2.4.0+cu121 torchvision==0.19.0+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install carvekit --no-deps
conda install -y -c conda-forge eigen
```

PyTorch3D is required by the rendering pipeline (`--render_method hybrid/mesh`). We recommend installing from source:

```bash
bash extern/install_pytorch3d.sh
```

Requirements for building PyTorch3D: CUDA toolkit with `nvcc`, and a C++ compiler (`g++` / `c++`).

By default, inference also uses `vipe` (for `--align_with_vda`). `vipe` builds its CUDA extension lazily on first import, so after the environment is ready we recommend validating it once:

```bash
python extern/vipe_test.py
```

This checks that `extern/vipe` is importable and that `vipe.ext` can be compiled / loaded successfully in the current environment.


### 3. Download pretrained models

By default, UniWorld-View loads weights from `./checkpoints/` (see `checkpoints/README.md`).

#### Option A (recommended): download to local `./checkpoints/`

```bash
bash checkpoints/download_hf.sh
```

This downloads: UniWorld-View transformer, Wan2.1-VACE, BLIP2, STream3R, MoGe, SAM2, TracerB7, Video-Depth-Anything (for `--align_with_vda`), and the default CausVid LoRA `v2`.

```

#### Option B: load from Hugging Face Hub (cache)

Run inference with `--no_load_weights_locally` and override repo IDs via environment variables or CLI (see `checkpoints/download_hf.py`).

## 💫 Inference

Run from the **repo root** (`UniWorld-View/`).

### 1. Command line

```bash
bash run_infer.sh
```

To run on a different GPU: `CUDA_DEVICE=3 bash run_infer.sh` or pass `--device cuda:3` after `run_infer.sh`.

### 2. Local Gradio demo

```bash
bash run_app.sh
```

Open `http://127.0.0.1:7860` in your browser. For remote access: `GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT=7860 bash run_app.sh`.

## 📢 Limitations

Our model excels at handling videos with well-defined objects and clear motion, as demonstrated in the demo videos. However, since it is built upon a pretrained video diffusion model, it may struggle with complex cases that go beyond the generation capabilities of the base model.

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
