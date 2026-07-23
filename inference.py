import os
import torch

from configs.infer_config import get_parser
from datetime import datetime

from demo import UniScene



if __name__=="__main__":
    parser = get_parser() # infer config.py
    opts = parser.parse_args()
    opts.save_dir = os.path.join(opts.out_dir, os.path.splitext(os.path.basename(opts.image_dir))[0])
    opts.weight_dtype = torch.bfloat16
    os.makedirs(opts.save_dir,exist_ok=True)
    
    # opts.device = torch.device(opts.device)
    pvd = UniScene(opts)

    if opts.mode == 'dynamic_view_pre':
        pvd.nvs_dynamic_view()
    
    elif opts.mode == 'align_depth':
        pvd.align_bg_depth()

    elif opts.mode == 'dynamic_view_left':
        pvd.gen_sup_views('left')
    
    elif opts.mode == 'dynamic_view_right':
        pvd.gen_sup_views('right')

    else:
        raise KeyError(f"Invalid Mode: {opts.mode}")
