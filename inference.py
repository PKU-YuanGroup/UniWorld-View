import os

from utils.device_compat import ensure_ascend_hint, resolve_device, setup_backend

setup_backend()
ensure_ascend_hint()

import torch

from configs.infer_config import get_parser
from datetime import datetime

from demo import UniScene



if __name__=="__main__":
    parser = get_parser() # infer config.py
    opts = parser.parse_args()
    if opts.exp_name == None:
        prefix = datetime.now().strftime("%Y%m%d_%H%M")
        opts.exp_name = f'{prefix}_{os.path.splitext(os.path.basename(opts.image_dir))[0]}'
    opts.save_dir = os.path.join(opts.out_dir,opts.exp_name)
    opts.weight_dtype = torch.bfloat16
    # Normalize device: allow cuda:N / npu:N / cpu
    if not opts.device or opts.device == "cuda:0":
        # Auto-pick NPU when Ascend is available and user left CUDA default.
        opts.device = resolve_device(None) if not os.environ.get("CUDA_VISIBLE_DEVICES") else opts.device
    # If user explicitly passed npu / cuda / cpu, keep it; only rewrite bare default via resolve when npu present.
    if str(opts.device).startswith("cuda") and hasattr(torch, "npu") and torch.npu.is_available() and not torch.cuda.is_available():
        # Running on Ascend box with legacy cuda:0 default/arg → map to npu index
        idx = "0"
        if ":" in str(opts.device):
            idx = str(opts.device).split(":", 1)[1] or "0"
        opts.device = f"npu:{idx}"
        print(f"[inference] Remapped device to {opts.device} (Ascend NPU, no CUDA)")
    os.makedirs(opts.save_dir,exist_ok=True)
    # opts.device = torch.device(opts.device)
    pvd = UniScene(opts)

    if opts.mode == 'single_view':
        pvd.nvs_single_view()

    elif opts.mode == 'dynamic_view':
        # UniScene already holds opts internally; avoid redundant passing
        pvd.nvs_dynamic_view()

    else:
        raise KeyError(f"Invalid Mode: {opts.mode}")
