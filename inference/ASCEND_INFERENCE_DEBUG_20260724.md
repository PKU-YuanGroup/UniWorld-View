# UniWorld-View Ascend NPU 推理调试记录

- **日期**: 2026-07-24
- **环境**: openEuler 22.03 / Ascend NPU（单卡约 61–64GB HBM） / conda 环境 `uniworld-view`
- **入口**: `bash run_infer.sh`（`--device npu:0`，`--mode dynamic_view`）
- **首次完整跑通输出目录**: `output/20260724_1956_2/`
- **关联日志**: `logs/run_infer_20260724_195617.log`

本文记录从权重加载失败到首次完整产出扩散结果的分析过程、问题与解决方法。更早的环境适配见仓库根目录 `ASCEND_NPU_ISSUES.md`。

---

## 1. 总体结论

在 Ascend NPU 上跑通 `dynamic_view` 推理，核心约束是：

1. **显存互斥**：Wan-14B diffusion pipeline 与 STream3R / SAM2 / BLIP2 等辅模型不能同时驻留 NPU。
2. **算子兼容**：RoPE 的 `complex128`、SAM2 默认 `cuda`、decord 与 FFmpeg 8 不兼容等需专门适配。
3. **依赖与渲染路径**：昇腾上优先 `--render_method warp`（避免 PyTorch3D/CUDA）；补齐 `av` / `ftfy` 等依赖。

当前可复现命令（推荐）：

```bash
cd /home/wd/wd_space/UniWorld/UniWorld-View
bash run_infer.sh --render_method warp
# run_infer.sh 已默认包含 --render_method warp / --no_align_with_vda / NPU 环境变量
```

> **注意**：调试过程中曾临时注释掉 `./checkpoints/UniView` 预训练权重加载（仅用 config 随机初始化 transformer）。若日志出现  
> `[SKIP] loading 3D transformer's pretrained weights from ./checkpoints/UniView`，  
> 扩散观感会明显变差。正式推理需恢复权重加载，并保证 `checkpoints/UniView` 含完整 safetensors 分片。

---

## 2. 问题与解决方法（按时间顺序）

### 问题 A：加载 UniView 权重 `KeyError: 'patch_embedding.weight'`

**现象**

```text
loaded 3D transformer's pretrained weights from ./checkpoints/UniView ...
KeyError: 'patch_embedding.weight'
位置: model/uniview_transformer.py → from_pretrained()
```

**原因分析**

- `checkpoints/UniView` 当时仅有 `config.json` 与 index，缺少实际 `.safetensors` 权重分片。
- `from_pretrained` 扫到空 `state_dict` 后仍访问 `state_dict['patch_embedding.weight']` 导致崩溃。

**解决方法（临时跳过，便于继续打通流水线）**

- 备份：`model/uniview_transformer.original_2.py`
- 修改：`model/uniview_transformer.py`  
  - 注释掉 checkpoint 权重加载逻辑；保留读 `config.json` + `from_config` 构建模型。
  - 打印改为 `[SKIP] loading ...`

**后续正式使用**

1. 补全权重：`HF_ENDPOINT=https://hf-mirror.com bash checkpoints/download_hf.sh`
2. 恢复 `uniview_transformer.py` 中权重加载代码（可用 `original_2` 对照）

---

### 问题 B：`decord` + FFmpeg 8 `pix_fmts` 报错

**现象**

```text
Option 'pix_fmts' is not a runtime option ...
DECORDError: ... filter_graph.cc ... Set output pixel format error.
```

**原因分析**

- conda 中 `ffmpeg=8.0.1` 与旧版 decord（conda-forge `0.6.0`）API 不兼容。
- FFmpeg 7+/8 上 buffersink 的 `pix_fmts` 不再是可在初始化后设置的 runtime option。

**解决方法**

1. 安装系统兼容 FFmpeg 4.2.4（openEuler 仓库）：
   ```bash
   dnf install -y ffmpeg ffmpeg-libs ffmpeg-devel
   ```
2. 从源码编译安装 decord，并链接系统 FFmpeg：
   - 源码：`/home/wd/wd_space/UniWorld/decord`
   - 前缀布局：`/home/wd/wd_space/UniWorld/ffmpeg-4.2` → `include`/`lib` 指向系统头文件与 `/usr/lib64`
   - cmake：`-DFFMPEG_DIR=.../ffmpeg-4.2 -DUSE_CUDA=0`
   - `python setup.py install`，并将 `libdecord.so` 放到：
     - `$CONDA_PREFIX/lib/python3.10/site-packages/decord/`
     - `$CONDA_PREFIX/lib/`
3. 验证：`VideoReader` 可读 `test/videos/2.mp4`（300 帧）。

**说明**：conda PATH 中仍可能是 ffmpeg 8 可执行文件；decord 通过 SONAME（`libavcodec.so.58` 等）绑定系统 FFmpeg 4.2，二者可并存。

---

### 问题 C：STream3R 阶段 `NPU out of memory`

**现象**

```text
RuntimeError: NPU out of memory. Tried to allocate 708.00 MiB
(... 58.12 GiB already allocated; 364.99 MiB free ...)
位置: stream3r 前向 / patch embedding
```

**原因分析**

- `__init__` 中 `setup_diffusion()` 把 Wan-14B pipeline 整机放到 NPU（约占用 ~58GB）。
- `nvs_dynamic_view()` 进入 `active_aux_modules()` 时又把 STream3R/MoGe/BLIP2/segnet 搬回 NPU。
- 两套大模型同时驻留 → OOM。

**解决方法**

- 备份：`demo.original_4.py`（后续还有 `original_5/6/8`）
- 修改：`demo.py`
  1. 增加 `offload_pipeline_to_cpu()` / `ensure_pipeline_on_device()`
  2. `active_aux_modules`：先卸 pipeline，再加载辅模型
  3. `aux_modules_offloaded`（diffusion 前）：先卸辅模型，再加载 pipeline
  4. STream3R 阶段仅保留 `stream3r` 在 NPU，推理后立刻卸回 CPU
  5. NPU 上 diffusion 默认放 CPU，按需搬上设备

---

### 问题 D：`UnboundLocalError: imgs_stream`

**现象**

```text
del imgs_stream, outputs
UnboundLocalError: local variable 'imgs_stream' referenced before assignment
```

**原因分析**

- 为省显存在 STream3R 后已 `del imgs_stream`，后面再次 `del imgs_stream, outputs` 触发异常。

**解决方法**

- 备份：`demo.original_5.py`
- 修改：后一处改为仅 `del outputs`

---

### 问题 E：`render_method=hybrid` 需要 PyTorch3D

**现象**

```text
ModuleNotFoundError: No module named 'pytorch3d'
render_method='hybrid' requires PyTorch3D ...
```

**原因分析**

- 默认 `--render_method hybrid` 依赖 PyTorch3D（通常需 CUDA/`nvcc`）。
- Ascend 环境难以安装官方 PyTorch3D。

**解决方法**

- 使用不依赖 PyTorch3D 的路径：`--render_method warp`
- `run_infer.sh` 默认加入 `--render_method warp`（备份：`run_infer.original_6.sh`）

---

### 问题 F：SAM2 默认加载到 `cuda`

**现象**

```text
AssertionError: Torch not compiled with CUDA enabled
位置: build_sam2_video_predictor → model.to(device) 且 device 默认为 "cuda"
```

**原因分析**

- `setup_sam2()` 未传 `device`，SAM2 构建默认 `device="cuda"`。
- 本环境为 torch_npu，未编译 CUDA。

**解决方法**

- 备份：`demo.original_6.py`
- 修改：`build_sam2_video_predictor(..., device=str(self.opts.device))`（如 `npu:0`）

**附带警告（可忽略）**

```text
cannot import name '_C' from 'sam2' ... Skipping the post-processing step
```

SAM2 C 扩展未编译，仅影响部分后处理，主路径仍可用。

---

### 问题 G：缺少 PyAV（`av`）

**现象**

```text
ImportError: PyAV is not installed ... torchvision.io.write_video
```

**原因分析**

- `utils/warp_utils.save_video` 调用 `torchvision.io.write_video`，依赖 PyAV。
- `requirements.txt` 含 `av==12.0.0`，环境未安装。

**解决方法**

```bash
pip install 'av==12.0.0'
```

---

### 问题 H：`NameError: name 'ftfy' is not defined`

**现象**

```text
File pipeline_uniview.py → basic_clean → ftfy.fix_text
NameError: name 'ftfy' is not defined
```

**原因分析**

- `ftfy` 仅在 `is_ftfy_available()` 为真时 import，但 `basic_clean` 无条件调用。
- 环境未装 `ftfy` 时触发 NameError。

**解决方法**

```bash
pip install ftfy
```

（`requirements.txt` 已列出；备份曾准备 `pipeline_uniview.original_7.py`）

---

### 问题 I：RoPE `complex128` 在 NPU 上不支持

**现象**

```text
RuntimeError: cat ... aclnnCat failed ... 207001
tensor 0 not implemented for DT_COMPLEX128
(... 支持列表含 DT_COMPLEX64，不含 COMPLEX128)
位置: uniview_transformer.py WanRotaryPosEmbed.forward / apply_rotary_emb
```

**原因分析**

- 原实现非 MPS 时用 `float64` → `view_as_complex` 得到 `complex128`。
- Ascend NPU 不支持 `complex128`，仅支持 `complex64` 等。
- 同时大尺寸 complex128 中间张量加重显存压力。

**解决方法**

- 备份：`model/uniview_transformer.original_7.py`
- 修改：`model/uniview_transformer.py`
  1. NPU/MPS 上 RoPE 使用 `float32` → `complex64`
  2. freqs 在 CPU 上拼好后一次性 `.to(device)`，避免 NPU 上对 complex128 做 `cat`

---

### 问题 J：VAE decode 阶段 NPU OOM（扩散 8/8 之后）

**现象**

```text
100%|██████████| 8/8 ...
然后:
RuntimeError: NPU out of memory. Tried to allocate 1.46 GiB
(... 58.60 GiB already allocated ...)
位置: pipeline_uniview.py → self.vae.decode(latents)
```

**原因分析**

- 去噪循环结束后 Wan-14B transformer 仍占满 HBM（~58GB+）。
- VAE decode（float32 + 多帧）再申请约 1.46GB 失败。
- 进度条 8/8 只表示 denoising 结束，decode/后处理在其后。

**解决方法**

- 备份：`model/pipeline_uniview.original_8.py`、`demo.original_8.py`
- 修改：
  1. `pipeline_uniview.py`：decode 前将 `transformer` / `text_encoder` 卸到 CPU，`empty_cache`，确保 `vae` 在设备上；释放 conditioning 等大激活。
  2. `demo.py`：`vae.enable_tiling()` 降低解码峰值显存。
- `run_infer.sh` 已加部分 NPU 显存相关环境变量（如 `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`）。

---

## 3. 关键文件与备份对照

| 修改文件 | 备份后缀 | 主要改动 |
|----------|----------|----------|
| `model/uniview_transformer.py` | `original_2`, `original_7` | 跳过 UniView 权重加载（临时）；RoPE complex64 |
| `demo.py` | `original_4`–`6`, `original_8` | pipeline/辅模型互斥卸载；SAM2 device；VAE tiling；`imgs_stream` del 修复 |
| `run_infer.sh` | `original_6` | `--render_method warp`；日志 tee；NPU alloc 环境变量 |
| `model/pipeline_uniview.py` | `original_7`, `original_8` | decode 前卸载 transformer/text_encoder |

依赖/系统侧（非源码备份）：

- 系统 FFmpeg 4.2.4 + 源码编译 decord
- `pip install av ftfy`（及此前 requirements 中其它包）

---

## 4. 当前推荐运行配置

`run_infer.sh` 关键项摘要：

- `source` Ascend / CANN `set_env.sh`
- `HF_ENDPOINT=https://hf-mirror.com`
- `VIPE_EXT_SKIP_JIT=1`
- `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` 等
- `--device npu:0`
- `--no_align_with_vda`（避开 vipe CUDA 路径）
- `--render_method warp`
- 日志：`logs/run_infer_<timestamp>.log`

示例：

```bash
bash run_infer.sh --render_method warp
# 若仍 OOM，可尝试降低分辨率或帧数，例如：
# bash run_infer.sh --height 384 --width 640 --video_length 49
```

---

## 5. 输出目录含义（以 `output/20260724_1956_2` 为例）

流水线：输入视频 → 深度/位姿 → SAM 分割 → 几何 warp 渲染 → 扩散生成 → 颜色校正 → 对比视频。

| 文件/目录 | 含义 |
|-----------|------|
| `input.mp4` | 处理后的输入视频 |
| `sam/masks.mp4` | SAM2 前景掩码 |
| `sam/painted_images.mp4` | 掩码叠加可视化 |
| `tmp_dir/` | SAM 读帧临时 JPEG（可删） |
| `render.mp4` | 新视角几何渲染条件视频 |
| `mask.mp4` | 可见性/空洞掩码（扩散条件） |
| `mask_occ.mp4` | 遮挡相关掩码（调试） |
| `mask_inconsis.mp4` | 不一致区域掩码（调试） |
| `diffusion_uncorrect.mp4` | 扩散原始输出 |
| `diffusion_correct.mp4` | 颜色校正后输出 |
| `diffusion.mp4` | 左：输入，右：校正后生成（建议优先查看） |

判断是否跑通：

1. 终端回到 shell 提示符且无新的 `ERROR`/`Traceback`
2. 输出目录出现 `diffusion.mp4` / `diffusion_correct.mp4` 等完整文件  
（仅看到 `8/8` 进度条不等于结束；其后还有 VAE decode 与存盘。）

---

## 6. 已知限制与待办

1. **UniView 权重**：若仍为 SKIP 加载，需补全 checkpoint 并恢复加载逻辑，否则生成质量不可信。
2. **PyTorch3D / hybrid、mesh 渲染**：Ascend 上暂用 `warp`。
3. **VDA / vipe**：默认关闭；依赖 CUDA JIT。
4. **SAM2 `_C` 扩展**：未编译，后处理受限但可跑。
5. **显存**：单卡 ~64GB 跑 Wan-14B 仍紧；保持组件级卸载，必要时降分辨率/帧数。
6. **`linalg.inv` 等算子**：可能 CPU fallback，有性能提示属正常。

---

## 7. 简要时间线

| 阶段 | 结果 |
|------|------|
| UniView 权重 KeyError | 临时 SKIP 加载 |
| decord + FFmpeg8 | 系统 FFmpeg4.2 + 源码 decord |
| STream3R OOM | pipeline/辅模型互斥卸载 |
| imgs_stream UnboundLocal | 修复二次 del |
| hybrid / pytorch3d | 改用 warp |
| SAM2 cuda | 传入 npu device |
| 缺 av / ftfy | pip 安装 |
| RoPE complex128 | 改用 complex64 |
| VAE decode OOM | decode 前卸 transformer + tiling |
| `output/20260724_1956_2` | 首次完整产出 |

---

*文档位置：`inference/ASCEND_INFERENCE_DEBUG_20260724.md`*
