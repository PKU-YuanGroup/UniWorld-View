## ___***UniWorld-View: 基于视频扩散模型的大基线视角合成***___
<div align="center">

 <a href='https://github.com/PKU-YuanGroup/UniWorld-View'><img src='https://img.shields.io/badge/GitHub-UniWorld-View-blue.svg'></a> &nbsp;
 <a href='https://huggingface.co/Drexubery/UniView'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a> &nbsp;

</div>

🤗 如果觉得 UniWorld-View 有用，**请帮忙给本仓库点 ⭐**，这对开源项目很重要。谢谢！

## 🔆 简介

UniWorld-View 可以从<strong>随意拍摄的单目视频</strong>或<strong>单张图像</strong>生成高保真新视角，同时支持高精度的位姿控制。

<!-- 有演示 GIF 或图片时可在此添加，例如：
<table class="center">
    <tr style="font-weight: bolder;">
        <td>输入视频 / 图像 &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; 新视角</td>
    </tr>
  <tr>
    <td>
    <img src=assets/demo.gif style="width: 100%; height: auto;">
  </td>
  </tr>
</table>
-->

如需进行 4D 重建，请切换到 `recon` 分支。

## ⚙️ 环境配置

### 0. GPU 显存要求

建议使用 **显存 ≥ 60GB** 的 GPU，以保证推理流畅。

### 1. 克隆本仓库

```bash
git clone https://github.com/PKU-YuanGroup/UniWorld-View.git
cd UniWorld-View
```

推理需要在 `extern/` 下准备以下第三方仓库：

```bash
git clone https://github.com/NIRVANALAN/STream3R.git extern/STream3R
git clone https://github.com/nv-tlabs/vipe.git extern/vipe
```

期望目录结构：

```text
UniWorld-View/
  extern/
    STream3R/
    vipe/
```

为使 `vipe` / VDA 在非默认 GPU 上正确支持 `--device cuda:N`，请用 `extern/vipe_patches/` 中附带的补丁文件替换官方 `vipe` 仓库中的对应两个文件：

```bash
cp extern/vipe_patches/videodepthanything/__init__.py \
  extern/vipe/vipe/priors/depth/videodepthanything/__init__.py
cp extern/vipe_patches/videodepthanything/video_depth.py \
  extern/vipe/vipe/priors/depth/videodepthanything/video_depth.py
```

### 2. 配置环境

```bash
conda create -n uniworld-view python=3.10 -y
conda activate uniworld-view
pip install torch==2.4.0+cu121 torchvision==0.19.0+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install carvekit --no-deps
conda install -y -c conda-forge eigen
```

渲染流程（`--render_method hybrid/mesh`）需要 PyTorch3D。建议从源码安装：

```bash
bash extern/install_pytorch3d.sh
```

编译 PyTorch3D 的要求：带有 `nvcc` 的 CUDA toolkit，以及 C++ 编译器（`g++` / `c++`）。

默认情况下，推理还会使用 `vipe`（用于 `--align_with_vda`）。`vipe` 会在首次 import 时惰性编译其 CUDA 扩展，因此环境就绪后建议先验证一次：

```bash
python extern/vipe_test.py
```

该脚本会检查 `extern/vipe` 是否可导入，以及当前环境中 `vipe.ext` 能否成功编译 / 加载。


### 3. 下载预训练模型

默认情况下，UniWorld-View 从 `./checkpoints/` 加载权重（详见 `checkpoints/README.md`）。

#### 方案 A（推荐）：下载到本地 `./checkpoints/`

```bash
bash checkpoints/download_hf.sh
```

将下载：UniWorld-View transformer、Wan2.1-VACE、BLIP2、STream3R、MoGe、SAM2、TracerB7、Video-Depth-Anything（用于 `--align_with_vda`），以及默认的 CausVid LoRA `v2`。

```

#### 方案 B：从 Hugging Face Hub 加载（缓存）

推理时加上 `--no_load_weights_locally`，并通过环境变量或 CLI 覆盖仓库 ID（详见 `checkpoints/download_hf.py`）。

## 💫 推理

请在**仓库根目录**（`UniWorld-View/`）下运行。

### 1. 命令行

```bash
bash run_infer.sh
```

在其他 GPU 上运行：`CUDA_DEVICE=3 bash run_infer.sh`，或在 `run_infer.sh` 后传入 `--device cuda:3`。

### 2. 本地 Gradio 演示

```bash
bash run_app.sh
```

在浏览器中打开 `http://127.0.0.1:7860`。远程访问：`GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT=7860 bash run_app.sh`。

## 📢 局限性

本模型在处理物体清晰、运动明确的视频时表现出色，如演示视频所示。但由于它基于预训练的视频扩散模型构建，对于超出基座模型生成能力的复杂场景，效果可能不佳。

## 🤗 相关工作

包括但不限于：[VACE](https://github.com/ali-vilab/VACE)、[ViewCrafter](https://github.com/Drexubery/ViewCrafter)、[GCD](https://gcd.cs.columbia.edu/)、[NVS-Solver](https://github.com/ZHU-Zhiyu/NVS_Solver)、[DimensionX](https://github.com/wenqsun/DimensionX)、[ReCapture](https://generative-video-camera-controls.github.io/)、[TrajAttention](https://xizaoqu.github.io/trajattn/)、[GS-DiT](https://wkbian.github.io/Projects/GS-DiT/)、[DaS](https://igl-hkust.github.io/das/)、[RecamMaster](https://github.com/KwaiVGI/ReCamMaster)、[GEN3C](https://research.nvidia.com/labs/toronto-ai/GEN3C/)、[CAT4D](https://cat-4d.github.io/)...

## 📜 引用

如果觉得本工作有帮助，请考虑引用：

```BibTeX
@misc{UniWorld-View2025,
    author    = {PKU-YuanGroup},
    title     = {UniWorld-View: Large-Baseline View Synthesis via Video Diffusion Models},
    year      = {2025},
    url       = {https://github.com/PKU-YuanGroup/UniWorld-View}
}
```

## 许可证

本项目采用 Apache-2.0 许可证。详见 `LICENSE`。
