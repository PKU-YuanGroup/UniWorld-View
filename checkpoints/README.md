# Checkpoints

This folder is **not tracked** by git (model weights can be very large).

## Quick start

```bash
bash checkpoints/download_hf.sh
```

By default this downloads **only the required modules**: the core UniView
pipeline (transformer, Wan2.1-VACE, BLIP2, MoGe, SAM2, tracer) and the
default CausVid LoRA.

## Optional module bundles

The script exposes three flags that opt-in to downloading an additional
bundle. All three default to `False`; pass them explicitly to pull each one.

| Flag          | Bundle                                                                       |
| ------------- | ---------------------------------------------------------------------------- |
| `--mosca`     | MoSca auxiliary bundle (DepthCrafter, SVD, Metric3D)                         |
| `--stream3r`  | STream3R auxiliary bundle (alternative to `--mosca`)                         |
| `--recon`     | Reconstruction bundle (ROSE + Wan2.1-Fun-1.3B-InP)                           |

Examples:

```bash
# Download only the required weights (default behavior).
bash checkpoints/download_hf.sh

# Download core weights and STream3R weights.
python checkpoints/download_hf.py --stream3r

# Download core weights and MoSca weights.
bash checkpoints/download_hf.sh --mosca

# Download MoSca + recon for 4D reconstruction.
bash checkpoints/download_hf.sh --mosca --recon
```

This downloads the default CausVid LoRA:

```text
checkpoints/loras/Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors
```

If you want the older `v1` LoRA instead, override the filename:

```bash
WAN_LORA_FILENAME=Wan21_CausVid_14B_T2V_lora_rank32.safetensors \
    bash checkpoints/download_hf.sh
```

Place downloaded weights here, e.g.:

- `checkpoints/UniView/` (main UniView transformer weights)
- `checkpoints/blip2-opt-2.7b/` (BLIP2 captioning model)
- `checkpoints/sam2/` (SAM2 weights + configs)
- `checkpoints/moge/` (MoGe weights)
- `checkpoints/tracer_b7.pth` (foreground tracer model)
- `checkpoints/depthcrafter/`, `checkpoints/stable-video-diffusion-img2vid-xt/`,
  `checkpoints/metric3d/` (MoSca bundle, requires `--mosca`)
- `checkpoints/STream3R/` (STream3R bundle, requires `--stream3r`)
- `checkpoints/rose/transformer/`, `checkpoints/rose/Wan2.1-Fun-1.3B-InP/`
  (reconstruction bundle, requires `--recon`)
- `checkpoints/loras/` (CausVid LoRA weights, `v2` recommended by default)

See `checkpoints/download_hf.py` for the full list of Hugging Face repos /
filenames and optional env vars.