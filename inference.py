from demo_dynamic_vipe import UniScene
import os
from configs.infer_config import get_parser
from datetime import datetime
import torch


if __name__=="__main__":
    parser = get_parser() # infer config.py
    opts = parser.parse_args()
    if opts.exp_name == None:
        prefix = datetime.now().strftime("%Y%m%d_%H%M")
        opts.exp_name = f'{prefix}_{os.path.splitext(os.path.basename(opts.image_dir))[0]}'
    opts.save_dir = os.path.join(opts.out_dir,opts.exp_name)
    opts.weight_dtype = torch.bfloat16
    os.makedirs(opts.save_dir,exist_ok=True)
    # opts.device = torch.device(opts.device)
    pvd = UniScene(opts)

    if opts.mode == 'single_view':
        pvd.nvs_single_view()

    elif opts.mode == 'sparse_view':
        pvd.nvs_sparse_view()

    elif opts.mode == 'dynamic_view':
        # UniScene already holds opts internally; avoid redundant passing
        pvd.nvs_dynamic_view()

    else:
        raise KeyError(f"Invalid Mode: {opts.mode}")
