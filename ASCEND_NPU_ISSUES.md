# UniWorld-View Ascend NPU 适配问题记录

日期: 2026-07-23

## 问题 1: `ModuleNotFoundError: huggingface_hub`
- 原因: 未安装 `requirements.txt`
- 处理: `pip install huggingface-hub==0.34.4` / 完整 `pip install -r requirements.txt`

## 问题 2: HuggingFace 下载慢
- 原因: 直连 huggingface.co
- 处理: `export HF_ENDPOINT=https://hf-mirror.com`（已写入 `run_infer.sh`）

## 问题 3: `ModuleNotFoundError: cv2` / `libGL.so.1`
- 原因: 未装 OpenCV；`opencv-python` 依赖 GUI 库
- 处理: `pip install opencv-python-headless==4.11.0.86`

## 问题 4: `ModuleNotFoundError: hydra`
- 原因: 未装 `hydra-core`
- 处理: `pip install hydra-core==1.3.2` / requirements

## 问题 5: `OSError: CUDA_HOME is not set`（vipe JIT）
- 原因: `demo.py` 顶部 import vipe → `vipe.ext` 强制 JIT 编译 NVIDIA CUDA 扩展；昇腾无 CUDA toolkit
- 处理:
  1. 修改 `extern/vipe/vipe/ext/__init__.py`：无 CUDA 时跳过 JIT
  2. `demo.py` 懒加载 VDA
  3. `export VIPE_EXT_SKIP_JIT=1`
  4. 默认 `--no_align_with_vda`（VDA/vipe CUDA 路径在 NPU 上暂不启用）

## 问题 6: `--align_with_vda False` 无效
- 原因: vipe 在 import 阶段就加载，与 CLI 开关无关
- 处理: 懒加载 + 默认关闭 VDA

## 问题 7: torch_npu 报 `libhccl.so` / `npu False`
- 原因: 未 source Ascend/CANN 环境
- 处理: `run_infer.sh` 中 source
  - `/usr/local/Ascend/ascend-toolkit/set_env.sh`
  - `/usr/local/Ascend/cann-9.0.0/set_env.sh`

## 问题 8: 代码硬编码 `cuda`
- 原因: 原项目为 NVIDIA GPU 开发
- 处理: 新增 `utils/device_compat.py`；`demo.py`/`inference.py`/`infer_config.py`/`run_infer.sh` 改为支持 `npu:N`

## 问题 9: 权重不完整
- 现象: `checkpoints/UniView` 仅有 config/index，缺 safetensors 分片；Wan/BLIP2/STream3R 等未下载
- 处理: 需重新执行 `HF_ENDPOINT=https://hf-mirror.com bash checkpoints/download_hf.sh`

## 问题 10: NPU 显存被 VLLM 占满
- 现象: `npu-smi` 显示 8 卡均有 `VLLMWorker_PP`，HBM ~32GB/64GB
- 影响: 大模型推理可能 OOM，需空闲 NPU 或停掉冲突进程

## 问题 11: `ModuleNotFoundError: diffusers` 等
- 原因: 环境未完整安装 requirements
- 处理: `pip install -r requirements.txt`
