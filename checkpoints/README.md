# Checkpoints

This folder is **not tracked** by git (model weights can be very large).

To download the required weights automatically:

```bash
bash checkpoints/download_hf.sh
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
- `checkpoints/vda/video_depth_anything_vitl.pth` (Video-Depth-Anything checkpoint for `--align_with_vda`)
- `checkpoints/tracer_b7.pth` (foreground tracer model)
- `checkpoints/loras/` (CausVid LoRA weights, `v2` recommended by default)

See `checkpoints/download_hf.py` for the list of Hugging Face repos / filenames and optional env vars.
