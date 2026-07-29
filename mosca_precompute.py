import torch
import imageio
import os, os.path as osp
import math
import numpy as np
import sys

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "extern/MoSca"))

from lib_prior.moca_processor import MoCaPrep
from lib_prior.preprocessor_utils import load_imgs, convert_from_mp4
from lib_prior.prior_loading import Saved2D, visualize_track
from lib_prior.moca_processor import mark_dynamic_region

from lib_render.render_helper import GS_BACKEND

from lib_moca.moca import moca_solve
from lib_moca.epi_helpers import analyze_track_epi, identify_tracks
from lib_moca.camera import MonocularCameras

from viz_utils import viz_list_of_colored_points_in_cam_frame
import logging
from lib_prior.moca_processor import *
from omegaconf import OmegaConf
from lib_moca.moca_misc import make_pair_list
import random


def seed_everything(seed):
    logging.info(f"seed: {seed}")
    print(f"seed: {seed}")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    logging.info(f"seed: {seed}")
    print(f"seed: {seed}")


def get_moca_processor(pre_cfg):
    moca_processor = MoCaPrep(
        dep_mode=getattr(
            pre_cfg, "dep_mode", "sensor"
        ),  # "depthcrafter", "metric3d", "uni"
        tap_mode=getattr(
            pre_cfg, "tap_mode", "bootstapir"
        ),  # "spatracker", "cotracker"
        flow_mode=getattr(pre_cfg, "flow_mode", "raft"),
        align_metric_flag=getattr(pre_cfg, "align_metric_flag", True),
        flow_ckpt=getattr(pre_cfg, "flow_ckpt", "./checkpoints/mosca/raft_models/raft-things.pth"),
        tap_ckpt=getattr(pre_cfg, "tap_ckpt", "./checkpoints/mosca/spaT_final.pth"),
        depthcrafter_unet_path=getattr(
            pre_cfg,
            "depthcrafter_unet_path",
            "tencent/DepthCrafter",
        ),
        depthcrafter_pretrain_path=getattr(
            pre_cfg,
            "depthcrafter_pretrain_path",
            "stabilityai/stable-video-diffusion-img2vid-xt",
        ),
        metric3d_ckpt_path=getattr(
            pre_cfg,
            "metric3d_ckpt_path",
            None,
        ),
        metric3d_torch_hub_path=getattr(
            pre_cfg,
            "metric3d_torch_hub_path",
            None,
        ),
    )
    return moca_processor


def load_imgs_from_dir(src):
    img_dir = osp.join(src, "images")
    img_fns = sorted(
        [it for it in os.listdir(img_dir) if it.endswith(".png") or it.endswith(".jpg")]
    )
    img_list = [imageio.imread(osp.join(img_dir, it))[..., :3] for it in img_fns]
    return img_list, img_fns

def svae_imgs_for_dir(
    src,
    ws,
    video_length: int = -1,
    stride: int = 1,
    max_res: int = 0,
    height: int | None = None,
    width: int | None = None,
    center_crop: bool = True,
):
    """
    Read a video file from `src`, decode+sampling+resize in one pass, and
    save the resulting frames as 5-digit-numbered PNG images under
    `ws/images/`. Returns (img_list, img_fns) in the same format as
    `load_imgs_from_dir` so the rest of the preprocessing pipeline can
    consume the result unchanged.

    Resolution policy:
        When an explicit (height, width) is given AND `center_crop=True`
        (default), decode at the minimal-cover resolution that preserves the
        source aspect ratio — short edge of output = input height, long edge
        scaled by the same factor. Save at that cover size WITHOUT
        center-cropping, so the saved frames are the same aspect ratio as the
        original video. Demo's geometry_backend=mosca then locks to this saved
        (H, W), giving byte-identical (or 1-px-off due to codec even-rounding)
        frames between mosca precompute and the inference-time read.

        Pass `center_crop=False` to fall back to a direct (potentially
        distorting) resize to the exact (height, width).

    Args:
        src: path to a video file readable by decord (e.g. .mp4).
        ws: working directory; the images are written to `<ws>/images/`.
        video_length: maximum number of frames to keep. -1 means "all".
        stride: sample every `stride`-th frame before truncation
                (1 = keep every frame).
        max_res: downsample so max(H, W) <= max_res. 0 = no limit.
                 Ignored if `height` and `width` are both provided.
        height: optional explicit target height (overrides max_res).
        width:  optional explicit target width  (overrides max_res).
        center_crop: when both `height` and `width` are given, decode at the
                minimal-cover resolution and SAVE at that resolution (no
                center-crop) so aspect ratio is preserved. Demo's mosca
                backend locks to this saved size. Use --no-center_crop for a
                direct resize.
    """
    if not osp.exists(src):
        raise FileNotFoundError(f"Video not found: {src}")

    try:
        from decord import VideoReader, cpu
    except ImportError as exc:
        raise ImportError(
            "decord is required by svae_imgs_for_dir. Install with "
            "`pip install decord`."
        ) from exc

    print(f"==> processing video: {src}")
    vid0 = VideoReader(src, ctx=cpu(0))
    first0 = vid0.get_batch([0]).asnumpy()
    original_height, original_width = first0.shape[1:3]
    print(
        f"==> original video shape: "
        f"{(len(vid0), original_height, original_width, first0.shape[-1])}"
    )

    target_w = int(width) if width is not None else None
    target_h = int(height) if height is not None else None
    have_explicit = target_w is not None and target_h is not None

    # 1) Decide decode resolution.
    if have_explicit and center_crop:
        # Minimal-cover decode that preserves aspect ratio. The output
        # covers the requested (h, w); the side that exceeded its target is
        # kept as-is (no center-crop) so aspect ratio of the source is
        # preserved end-to-end.
        if original_width <= 0 or original_height <= 0:
            decode_w, decode_h = target_w, target_h
        else:
            scale_cover_w = float(target_w) / float(original_width)
            scale_cover_h = float(target_h) / float(original_height)
            scale = max(scale_cover_w, scale_cover_h)
            decode_w = int(math.ceil(float(original_width) * scale))
            decode_h = int(math.ceil(float(original_height) * scale))
    elif have_explicit:
        # Direct (potentially distorting) resize to (target_h, target_w).
        decode_w, decode_h = target_w, target_h
    else:
        max_res_i = int(max_res) if max_res is not None else 0
        if max_res_i <= 0:
            max_res_i = 1024
        scale = min(1.0, float(max_res_i) / float(max(original_height, original_width)))
        decode_h = max(16, int(round(original_height * scale)))
        decode_w = max(16, int(round(original_width * scale)))

    # Round decode sizes UP to a multiple of 16. Downstream networks
    # (VAE encoder, flow / TAP / depth backbones) all have stride-2
    # conv stacks reaching stride 16 and require spatial dims divisible
    # by 16 — even (multiple of 2) sizes are not enough.
    decode_w = max(16, ((decode_w + 15) // 16) * 16)
    decode_h = max(16, ((decode_h + 15) // 16) * 16)

    vid = VideoReader(src, ctx=cpu(0), width=decode_w, height=decode_h)

    # 2) Pick frame indices via stride + truncation.
    frames_idx = list(range(0, len(vid), int(stride)))
    if video_length != -1 and video_length < len(frames_idx):
        frames_idx = frames_idx[:video_length]
    print(
        f"==> downsampled shape: "
        f"{(len(frames_idx), decode_h, decode_w, first0.shape[-1])}, "
        f"with stride: {stride}"
    )
    print(f"==> final processing shape: {(len(frames_idx), decode_h, decode_w)}")

    if len(frames_idx) == 0:
        raise RuntimeError(f"No frames sampled from {src}")

    frames_np = vid.get_batch(frames_idx).asnumpy()  # uint8 (T, H, W, C)

    # 3) Save to disk as PNG and return list/fns for downstream use.
    #    With center_crop=True, frames_np is already at the cover resolution
    #    (short edge = target_h, long edge preserves aspect ratio). We do
    #    NOT center-crop — that would defeat the purpose and the saved
    #    (H, W) would no longer match demo's read_video_frames output.
    img_dir = osp.join(ws, "images")
    os.makedirs(img_dir, exist_ok=True)

    img_list = []
    img_fns = []
    for idx, frame in enumerate(frames_np):
        frame_rgb = frame[..., :3]  # drop alpha, matches load_imgs_from_dir
        fn = f"{idx:05d}.png"
        imageio.imwrite(osp.join(img_dir, fn), frame_rgb)
        img_list.append(frame_rgb)
        img_fns.append(fn)

    return img_list, img_fns


def load_imgs_from_mp4():
    raise RuntimeError("Not implemented yet")
    return


def preprocess(
    img_list: list,
    img_fns: list,
    ws: str,
    moca_processor: MoCaPrep,
    pre_cfg: OmegaConf,
    resample_for_dynamic=True,
):
    seed_everything(getattr(pre_cfg, "seed", 12345))
    start_t = time.time()
    logging.info("*" * 20 + " Preprocessing " + "*" * 20)
    logging.info(f"Working on {ws}, start phase-1 preprocessing")
    logging.info("*" * 20 + " Preprocessing " + "*" * 20)

    BOUNDARY_EHNAHCE_TH = getattr(pre_cfg, "boundary_enhance_th", -1)
    DEPTH_DIR_POSTFIX = "_depth_sharp" if BOUNDARY_EHNAHCE_TH > 0 else "_depth"

    EPI_TH = getattr(pre_cfg, "epi_th", 1e-3)
    DEPTH_BOUNDARY_TH = getattr(
        pre_cfg, "depth_boundary_th", 1.0
    )  # this is in the median=1.0 space

    TAP_CHUNK_SIZE = getattr(pre_cfg, "tap_chunk_size", 5000)

    moca_processor.process(
        t_list=None,
        img_list=img_list,
        img_name_list=img_fns,
        save_dir=ws,
        n_track=getattr(pre_cfg, "n_track_uniform", 8192),
        # depth crafter
        depthcrafter_denoising_steps=getattr(
            pre_cfg, "depthcrafter_denoising_steps", 25
        ),
        metric_alignment_frames=getattr(pre_cfg, "metric_alignment_frames", 10),
        metric_alignment_first_quantil=getattr(
            pre_cfg, "metric_alignment_first_quantil", 0.7
        ),
        metric_alignment_bias_flag=getattr(pre_cfg, "metric_alignment_bias_flag", True),
        metric_alignment_kernel=getattr(pre_cfg, "metric_alignment_kernel", "cauchy"),
        metric_alignment_fscale=getattr(pre_cfg, "metric_alignment_fscale", 0.001),
        # TAP
        compute_tap=True,
        tap_chunk_size=TAP_CHUNK_SIZE,
        # Flow
        flow_steps=getattr(pre_cfg, "flow_steps", [1, 3]),
        epi_num_threads=getattr(pre_cfg, "epi_num_threads", 64),
        # Dep enhance for spatracker
        boundary_enhance_th=BOUNDARY_EHNAHCE_TH,  # if > 0 will create a sharp dir
        # boost
        compute_flow=getattr(pre_cfg, "compute_flow", True),
    )

    if not resample_for_dynamic:
        duration = (time.time() - start_t) / 60.0
        logging.info(
            f"Preprocessing done, SKIP DYN RESAMPLE! time cost: {duration:.3f}min"
        )
        return

    logging.info("*" * 20 + " Preprocessing " + "*" * 20)
    logging.info(f"Working on {ws}, start phase-2 preprocessing, densify the fg TAP")

    s2d = (
        Saved2D(ws)
        .load_epi()
        .load_dep(f"{moca_processor.dep_mode}{DEPTH_DIR_POSTFIX}", DEPTH_BOUNDARY_TH)
        .normalize_depth(median_depth=1.0)
        .recompute_dep_mask(depth_boundary_th=DEPTH_BOUNDARY_TH)
        .load_track(f"*uniform*{moca_processor.tap_mode}", min_valid_cnt=4)
        .load_vos()
    )

    if hasattr(s2d, "epi"):
        sample_mask = s2d.epi > EPI_TH
    else:
        continuous_pair_list = make_pair_list(s2d.T, interval=[1, 4], dense_flag=True)
        F_list, epierr_list, _ = analyze_track_epi(
            continuous_pair_list, s2d.track, s2d.track_mask, H=s2d.H, W=s2d.W
        )
        track_static_selection, _ = identify_tracks(epierr_list, EPI_TH)
        sample_mask = mark_dynamic_region(
            s2d.track[:, ~track_static_selection],
            s2d.track_mask[:, ~track_static_selection],
            s2d.H,
            s2d.W,
            0.1,
        )
    resampling_mask_dilate_ksize = getattr(pre_cfg, "resampling_mask_dilate_ksize", 7)
    sample_mask = (
        torch.nn.functional.max_pool2d(
            sample_mask[:, None].float(),
            kernel_size=resampling_mask_dilate_ksize,
            stride=1,
            padding=(resampling_mask_dilate_ksize - 1) // 2,
        )[:, 0]
        > 0.5
    )
    imageio.mimsave(
        osp.join(ws, "epi_resample_mask.gif"),
        sample_mask.cpu().numpy().astype(np.uint8) * 255,
    )

    moca_processor.compute_tap(
        ws=ws,
        save_name=f"dynamic_dep={moca_processor.dep_mode}",
        # n_track=8192 * 3,
        n_track=getattr(pre_cfg, "n_track_pdynamic", 8192 * 3),
        img_list=img_list,
        mask_list=sample_mask.detach().cpu().numpy() > 0,
        dep_list=moca_processor.load_dep_list(
            ws, f"{moca_processor.dep_mode}{DEPTH_DIR_POSTFIX}"
        ),
        # K=cams.default_K.detach().cpu().numpy(), # ! maintain the same K as the first infered static one
        max_viz_cnt=getattr(pre_cfg, "max_viz_cnt", 512),
        chunk_size=TAP_CHUNK_SIZE,
    )

    duration = (time.time() - start_t) / 60.0
    logging.info(f"Preprocessing done, time cost: {duration:.3f}min")
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("MoSca-V2 Preprocessing")
    parser.add_argument(
        "--video_path",
        type=str,
        help="Source video path",
        required=True,
    )
    parser.add_argument("--ws", type=str, help="Source folder", required=True)
    parser.add_argument("--cfg", type=str, help="profile yaml file path", required=True)
    parser.add_argument(
        "--skip_dynamic_resample", action="store_true", help="skip dynamic resample"
    )
    parser.add_argument(
        "--video_length",
        type=int,
        default=-1,
        help="Max frames to keep (-1 = all). Passed to svae_imgs_for_dir.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Sample every Nth frame from the video. Passed to svae_imgs_for_dir.",
    )
    parser.add_argument(
        "--max_res",
        type=int,
        default=0,
        help=(
            "Downsample so max(H,W) <= max_res before writing PNGs "
            "(0 = default 1024). Ignored if --height/--width are both set."
        ),
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Optional explicit target height for decoded frames.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Optional explicit target width for decoded frames.",
    )
    parser.add_argument(
        "--from_images",
        action="store_true",
        help=(
            "Skip video decoding and read PNGs from <ws>/images/ directly "
            "(legacy behaviour, useful when frames are already extracted)."
        ),
    )
    parser.add_argument(
        "--center_crop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When --height and --width are both set, decode at a "
            "minimal-cover resolution and center-crop to (height, width) "
            "instead of directly resizing (preserves aspect ratio). "
            "Matches demo.py's read_video_frames(center_crop=True). "
            "Use --no_center_crop to fall back to direct resize."
        ),
    )
    args, unknown = parser.parse_known_args()

    if args.from_images:
        img_list, img_fns = load_imgs_from_dir(args.ws)
    else:
        img_list, img_fns = svae_imgs_for_dir(
            args.video_path,
            args.ws,
            video_length=args.video_length,
            stride=args.stride,
            max_res=args.max_res,
            height=args.height,
            width=args.width,
            center_crop=args.center_crop,
        )
    prep_cfg = OmegaConf.load(args.cfg)
    cli_cfg = OmegaConf.from_dotlist([arg.lstrip('--') for arg in unknown])
    prep_cfg = OmegaConf.merge(prep_cfg, cli_cfg)

    moca_processor = get_moca_processor(prep_cfg)

    preprocess(
        img_list=img_list,
        img_fns=img_fns,
        ws=args.ws,
        moca_processor=moca_processor,
        pre_cfg=prep_cfg,
        resample_for_dynamic=not args.skip_dynamic_resample,
    )
