# Checkpoints

This folder is **not tracked** by git (model weights can be very large).

To download the required weights automatically:

```bash
bash checkpoints/download_hf.sh
```

Place downloaded weights here, e.g.:

- `checkpoints/UniView/` (main UniView transformer weights)
- `checkpoints/blip2-opt-2.7b/` (BLIP2 captioning model)
- `checkpoints/sam2/` (SAM2 weights + configs)
- `checkpoints/moge/` (MoGe weights)
- `checkpoints/vda/video_depth_anything_vitl.pth` (Video-Depth-Anything checkpoint for `--align_with_vda`)
- `checkpoints/tracer_b7.pth` (foreground tracer model)
- `checkpoints/loras/` (LoRA weights, if used)

- `checkpoints/mosca/` (Optical flow and tracker weights for MoSca 4D reconstruction)
- `checkpoints/rose/` (ROSE weights for video inpainting)

See `checkpoints/download_hf.py` for the list of Hugging Face repos / filenames and optional env vars.
