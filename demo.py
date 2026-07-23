import sys
import os
from pathlib import Path
from typing import Tuple, List
from omegaconf import OmegaConf
from contextlib import contextmanager
import json
from huggingface_hub import hf_hub_download, snapshot_download
import gc
# Renamed from demo_dynamic_stream3r_vda_mask.py

# Make repo-local modules importable even when launched outside repo root.
# NOTE: Keep repo root at sys.path[0] to avoid name collisions (e.g. MoGe ships a top-level `utils` package).
_REPO_ROOT = Path(__file__).parent.resolve()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Make extern deps importable (after repo root).
for _rel in ("extern", "extern/ROSE", "extern/STream3R", "extern/MoSca", "extern/sam", "extern/sam2"):
    _p = _REPO_ROOT / _rel
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(1, str(_p))

# sam2
from mask_painter import mask_painter as mask_painter2
from painter import mask_painter, point_painter
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2_video_predictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# depth alignment
from lib_prior.depth_models.depth_utils import viz_depth_list, save_depth_list
from vipe.priors.depth.videodepthanything import VideoDepthAnythingDepthModel
from vipe.priors.depth import DepthEstimationInput
from vipe.utils.misc import unpack_optional
from vipe.priors.depth.alignment import align_inv_depth_to_depth

# diffusion packages
from diffusers import AutoencoderKLWan
from model.pipeline_uniview import WanVACEPipeline
from model.uniview_transformer import WanVACETransformer3DModel
from diffusers import FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler
from transformers import AutoTokenizer, UMT5EncoderModel

# blip packages
from transformers import AutoProcessor, Blip2ForConditionalGeneration

# STream3R
from stream3r.models.stream3r import STream3R
from stream3r.models.components.utils.pose_enc import pose_encoding_to_extri_intri
from stream3r.models.components.utils.geometry import closed_form_inverse_se3
from stream3r.stream_session import StreamSession
from utils.tensor_stream import _preprocess_frames_for_stream3r
from stream3r.utils.visual_utils import predictions_to_glb

# MoSca
from lib_prior.depth_models.depth_utils import viz_depth_list, save_depth_list

# ROSE inpainting
from extern.ROSE.inference import video_inpainting

# single view packages
from carvekit.ml.wrap.tracer_b7 import TracerUniversalB7
from diffusers.utils import export_to_video

from utils.utils import traj_map, points_padding, np_points_padding, set_initial_camera, build_cameras
from datetime import datetime
from extern.moge.model.v2 import MoGeModel  # MoGe-2

from utils.warp_utils import *  # read_video_frames, save_video, etc.
import hydra
from utils.imagesplatrender import ImageSplattingRenderer

# other packages
import shutil
import einops
import warnings
import torch
import numpy as np
import torchvision
import copy
import cv2
from PIL import Image
from torchvision.utils import save_image
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from torchvision.transforms import ToTensor, ToPILImage
from pytorch3d.transforms import quaternion_to_matrix


CACHE_ROOT = "/mnt/data/ywb/cache_ckpts"
os.environ["HF_HOME"] = f"{CACHE_ROOT}/huggingface"
os.environ["TRANSFORMERS_CACHE"] = f"{CACHE_ROOT}/huggingface/transformers"
os.environ["HF_DATASETS_CACHE"] = f"{CACHE_ROOT}/huggingface/datasets"
os.environ["HF_HUB_CACHE"] = f"{CACHE_ROOT}/huggingface/hub"
os.environ["TORCH_HOME"] = f"{CACHE_ROOT}/torch"
os.environ["XDG_CACHE_HOME"] = CACHE_ROOT



class UniScene:
    def __init__(self, opts, gradio: bool = False):
        self.opts = opts
        self.device = opts.device
        self._apply_weight_source_policy
        
        if not hasattr(self.opts, "base_height"):
            self.opts.base_height = int(getattr(self.opts, "height", 480))
        if not hasattr(self.opts, "base_width"):
            self.opts.base_width = int(getattr(self.opts, "width", 832))
            
        #self.setup_stream3r()
        self.setup_moge()
        #self.setup_diffusion()

        self.caption_processor = AutoProcessor.from_pretrained(opts.blip_path)
        self.captioner = Blip2ForConditionalGeneration.from_pretrained(opts.blip_path, torch_dtype=torch.float16).to(opts.device)
        self.renderer = ImageSplattingRenderer(resolution=(opts.height, opts.width), device=opts.device)
        #self.offload_aux_modules_to_cpu()

    def _diffusion_spatial_base(self) -> int:
        """Pixel-space divisibility required by VAE downsample + transformer patch size."""
        try:
            vae_scale_factor = int(getattr(self.pipeline, "vae_scale_factor_spatial", 8))
            patch_size = getattr(self.pipeline.transformer.config, "patch_size", (1, 2, 2))
            return vae_scale_factor * int(patch_size[1])
        except Exception:
            return 16

    def _compute_target_hw_keep_aspect_ratio(self, orig_h: int, orig_w: int) -> tuple[int, int]:
        base = self._diffusion_spatial_base()
        base_h = int(getattr(self.opts, "base_height", self.opts.height))
        base_w = int(getattr(self.opts, "base_width", self.opts.width))
        token_budget = max(1, (base_h // base) * (base_w // base))

        if orig_h <= 0 or orig_w <= 0:
            return max(base, base_h // base * base), max(base, base_w // base * base)

        ratio = float(orig_w) / float(orig_h)
        w_center = max(1, int(round(math.sqrt(token_budget * ratio))))
        h_center = max(1, int(round(math.sqrt(token_budget / ratio))))

        best_wt, best_ht = w_center, h_center
        best_score = float("inf")
        for wt in range(max(1, w_center - 4), w_center + 5):
            for ht in range(max(1, h_center - 4), h_center + 5):
                tokens = wt * ht
                ratio_err = abs((wt / ht) - ratio)
                token_err = abs(tokens - token_budget)
                score = token_err + ratio_err * token_budget
                if score < best_score:
                    best_score = score
                    best_wt, best_ht = wt, ht

        target_w = best_wt * base
        target_h = best_ht * base

        max_res = int(getattr(self.opts, "max_res", 0) or 0)
        if max_res > 0 and max(target_w, target_h) > max_res:
            scale = max_res / float(max(target_w, target_h))
            target_w = max(base, int(round((target_w * scale) / base)) * base)
            target_h = max(base, int(round((target_h * scale) / base)) * base)

        return int(target_h), int(target_w)

    def _set_resolution(self, height: int, width: int) -> None:
        height = int(height)
        width = int(width)
        if int(getattr(self.opts, "height", height)) == height and int(getattr(self.opts, "width", width)) == width:
            return
        self.opts.height = height
        self.opts.width = width
        self.renderer = BiSplatRenderer(resolution=(height, width), device=self.opts.device)

    def _apply_input_resize_policy(self, image: Image.Image) -> Image.Image:
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL.Image.Image, got {type(image)}")

        image = ImageOps.exif_transpose(image)
        orig_w, orig_h = image.size

        if bool(getattr(self.opts, "keep_aspect_ratio", False)):
            target_h, target_w = self._compute_target_hw_keep_aspect_ratio(orig_h, orig_w)
            self._set_resolution(target_h, target_w)
            return image.resize((target_w, target_h), Image.Resampling.BICUBIC)
        else:
            base = self._diffusion_spatial_base()
            base_h = int(getattr(self.opts, "base_height", self.opts.height))
            base_w = int(getattr(self.opts, "base_width", self.opts.width))
            target_h = max(base, (base_h // base) * base)
            target_w = max(base, (base_w // base) * base)

        self._set_resolution(target_h, target_w)
        if (orig_w, orig_h) == (target_w, target_h):
            return image

        # Match SceneCrafter static preprocessing when not keeping aspect ratio:
        # 1) Resize the long edge to ~1024 (preserving aspect ratio).
        # 2) Center-crop to (target_w, target_h) without distortion.
        long_edge = 1024
        if orig_w <= 0 or orig_h <= 0:
            return image.resize((target_w, target_h), Image.Resampling.BICUBIC)

        if orig_w >= orig_h:
            new_w = long_edge
            new_h = max(1, int(round(orig_h * (float(long_edge) / float(orig_w)))))
        else:
            new_h = long_edge
            new_w = max(1, int(round(orig_w * (float(long_edge) / float(orig_h)))))

        image = image.resize((int(new_w), int(new_h)), Image.Resampling.BICUBIC)
        image = ImageOps.fit(image, (target_w, target_h), method=Image.Resampling.BICUBIC, centering=(0.5, 0.5))
        return image

    def _apply_weight_source_policy(self) -> None:
        """
        If `--no_load_weights_locally` is set, prefer Hugging Face Hub for pretrained weights
        (while still allowing users to override paths explicitly).
        """
        if bool(getattr(self.opts, "load_weights_locally", True)):
            return

        def _is_default_local_path(p: object) -> bool:
            if not isinstance(p, str):
                return False
            return p.startswith("checkpoints/") or p.startswith("./checkpoints/")

        # Repo IDs (can be overridden by explicitly passing corresponding args).
        hf_transformer_repo = os.environ.get("UNIVIEW_TRANSFORMER_REPO", "Drexubery/UniView")
        hf_wan_repo = os.environ.get("WAN_VACE_REPO", "Wan-AI/Wan2.1-VACE-14B-diffusers")
        hf_blip_repo = os.environ.get("BLIP2_REPO", "Salesforce/blip2-opt-2.7b")
        hf_stream3r_repo = os.environ.get("STREAM3R_REPO", "yslan/STream3R")
        hf_moge_repo = os.environ.get("MOGE_REPO", "Ruicheng/moge-2-vitl-normal")

        if not _is_default_local_path(getattr(self.opts, "transformer_path", "")):
            self.opts.transformer_path = hf_transformer_repo
        if not _is_default_local_path(getattr(self.opts, "model_name", "")):
            self.opts.model_name = hf_wan_repo
        if not _is_default_local_path(getattr(self.opts, "blip_path", "")):
            self.opts.blip_path = hf_blip_repo
        if not _is_default_local_path(getattr(self.opts, "stream3r_path", "")):
            self.opts.stream3r_path = hf_stream3r_repo
        if not _is_default_local_path(getattr(self.opts, "moge_path", "")):
            self.opts.moge_path = hf_moge_repo

        # File-based weights: download to HF cache and pass the resolved local path.
        segnet_path = getattr(self.opts, "segnet_path", None)
        if isinstance(segnet_path, str) and _is_default_local_path(segnet_path) and not Path(segnet_path).exists():
            self.opts.segnet_path = hf_hub_download(
                repo_id=os.environ.get("TRACER_REPO", "Carve/tracer_b7"),
                filename=os.environ.get("TRACER_FILENAME", "tracer_b7.pth"),
            )

        sam2_ckpt = getattr(self.opts, "sam2_checkpoint", None)
        if isinstance(sam2_ckpt, str) and _is_default_local_path(sam2_ckpt) and not Path(sam2_ckpt).exists():
            self.opts.sam2_checkpoint = hf_hub_download(
                repo_id=os.environ.get("SAM2_REPO", "facebook/sam2-hiera-large"),
                filename=os.environ.get("SAM2_FILENAME", "sam2_hiera_large.pt"),
            )

        lora_path = getattr(self.opts, "lora_path", None)
        if isinstance(lora_path, str) and _is_default_local_path(lora_path) and not Path(lora_path).exists():
            lora_repo = os.environ.get("WAN_LORA_REPO", "")
            lora_filename = os.environ.get("WAN_LORA_FILENAME", Path(lora_path).name)
            if lora_repo:
                self.opts.lora_path = hf_hub_download(repo_id=lora_repo, filename=lora_filename)
            else:
                self.opts.lora_path = None
        
        rose_model = getattr(self.opts, "rose_model", None)
        if isinstance(rose_model, str) and _is_default_local_path(rose_model) and not Path(rose_model).exists():
            rose_model = snapshot_download(
                repo_id=os.environ.get("WAN_INP_REPO", "alibaba-pai/Wan2.1-Fun-1.3B-InP"),
            )

        rose_ckpt = getattr(self.opts, "rose_ckpt", None)
        if isinstance(rose_ckpt, str) and _is_default_local_path(rose_ckpt) and not Path(rose_ckpt).exists():
            rose_ckpt = snapshot_download(
                repo_id=os.environ.get("ROSE_REPO", "Kunbyte/ROSE"),
            )
                
    def _safe_move_module(self, module, device):
        if module is None:
            return
        if isinstance(device, torch.device):
            device_str = device.type if device.type != "cuda" else f"cuda:{device.index or 0}"
        else:
            device_str = str(device)
        try:
            if hasattr(module, "to"):
                module.to(device_str)
            elif device_str.startswith("cuda") and hasattr(module, "cuda"):
                module.cuda()
            elif device_str == "cpu" and hasattr(module, "cpu"):
                module.cpu()
        except RuntimeError as exc:
            warnings.warn(f"Failed to move module {type(module).__name__} to {device_str}: {exc}")

    def offload_aux_modules_to_cpu(self):
        targets = [
            getattr(self, "stream3r", None),
            getattr(self, "depth_model", None),
            getattr(self, "sampredictor", None),
            getattr(self, "captioner", None),
            getattr(self, "seg_net", None),
        ]
        for module in targets:
            self._safe_move_module(module, "cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    def ensure_aux_modules_on_device(self, device: torch.device | str | None = None):
        target_device = device or self.opts.device
        targets = [
            getattr(self, "stream3r", None),
            getattr(self, "depth_model", None),
            getattr(self, "sampredictor", None),
            getattr(self, "captioner", None),
            getattr(self, "seg_net", None),
        ]
        for module in targets:
            self._safe_move_module(module, target_device)

    @contextmanager
    def active_aux_modules(self, device: torch.device | str | None = None):
        self.ensure_aux_modules_on_device(device)
        try:
            yield
        finally:
            self.offload_aux_modules_to_cpu()

    @contextmanager
    def aux_modules_offloaded(self, device: torch.device | str | None = None):
        self.offload_aux_modules_to_cpu()
        try:
            yield
        finally:
            self.ensure_aux_modules_on_device(device)
    def setup_stream3r(self):
        # Load pretrained STream3R once
        self.stream3r = STream3R.from_pretrained(self.opts.stream3r_path).to(self.opts.device).eval()
        self.session = StreamSession(self.stream3r, mode="causal")

    def setup_sam2(self):
        try:
            import iopath  # noqa: F401
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Missing dependency 'iopath' required by SAM2. "
                "Install with: pip install iopath==0.1.10 portalocker==2.8.2"
            ) from exc
        checkpoint = getattr(self.opts, "sam2_checkpoint", "./checkpoints/sam2/sam2_hiera_large.pt")
        model_cfg = getattr(self.opts, "sam2_config", "configs/sam2/sam2_hiera_l.yaml")
        self.sampredictor = build_sam2_video_predictor(model_cfg, checkpoint)
        self._safe_move_module(self.sampredictor, self.opts.device)
        return self.sampredictor
    
    def run_sam2_video(self, video, state, points, labels, bboxes):
        '''
        video: np.ndarray
        points: np.ndarray
        labels: np.ndarray
        '''
        mask_color = 3
        mask_alpha = 0.7
        contour_color = 1
        contour_width = 5
        point_color_ne = 8
        point_color_ps = 50
        point_alpha = 0.9
        point_radius = 15
        contour_color = 2
        contour_width = 5
        sam_masks = []
        sam_painted_images = []

        for idx in range(labels.shape[0]):
            # add new prompts and instantly get the output on the same frame
            frame_idx, object_ids, masks = self.sampredictor.add_new_points_or_box(state,frame_idx=0,obj_id=idx+1, points=points, labels=labels, box=bboxes)

        for out_frame_idx, out_obj_ids, out_mask_logits in self.sampredictor.propagate_in_video(state):
            sam_mask = (torch.sum(out_mask_logits > 0.0, dim=0)[0] > 0.0).cpu().numpy()
            painted_image = mask_painter(video[out_frame_idx], sam_mask.astype('uint8'), mask_color, mask_alpha, contour_color, contour_width)
            sam_masks.append(sam_mask)
            sam_painted_images.append(painted_image)
        print('sam_painted_images', len(sam_painted_images))

        return sam_masks, sam_painted_images

    def setup_moge(self):
        self.depth_model = MoGeModel.from_pretrained(self.opts.moge_path).to(self.opts.device).eval()

    def setup_diffusion(self):
        transformer = WanVACETransformer3DModel.from_pretrained(self.opts.transformer_path, torch_dtype=self.opts.weight_dtype)
        vae = AutoencoderKLWan.from_pretrained(self.opts.model_name, subfolder="vae", torch_dtype=torch.float32)

        tokenizer = AutoTokenizer.from_pretrained(self.opts.model_name, subfolder="tokenizer")
        text_encoder = UMT5EncoderModel.from_pretrained(
            self.opts.model_name, subfolder="text_encoder", low_cpu_mem_usage=True, torch_dtype=self.opts.weight_dtype
        ).eval()

        scheduler_config = {
            "num_train_timesteps": 1000,
            "shift": 5.0,
            "use_dynamic_shifting": False,
            "base_shift": 0.5,
            "max_shift": 1.15,
            "base_image_seq_len": 256,
            "max_image_seq_len": 4096,
        }
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

        self.pipeline = WanVACEPipeline(
            transformer=transformer,
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            scheduler=scheduler,
        )

        device_str = str(getattr(self.opts, "device", "cuda:0"))
        if device_str.startswith("cuda") and bool(getattr(self.opts, "low_gpu_memory_mode", False)):
            m = re.match(r"^cuda(?::(\d+))?$", device_str)
            gpu_id = int(m.group(1)) if m and m.group(1) is not None else 0
            self.pipeline.enable_model_cpu_offload(gpu_id=gpu_id)
        else:
            self.pipeline.to(device_str)

        lora_path = getattr(self.opts, "lora_path", None)
        if isinstance(lora_path, str) and lora_path:
            if Path(lora_path).exists():
                self.pipeline.load_lora_weights(lora_path, adapter_name="causvid_lora")
                self.pipeline.set_adapters(["causvid_lora"], adapter_weights=[0.95])
                self.pipeline.fuse_lora()
            else:
                warnings.warn(
                    f"LoRA weights not found at '{lora_path}'. "
                    "Proceeding without LoRA. (You can download weights to ./checkpoints/loras/ "
                    "or pass a valid --lora_path.)"
                )

    def run_moge(self, depth_image):
        with torch.no_grad():
            output = self.depth_model.infer(depth_image)
            depth = output["depth"]  # Depth in [m]. 
            normal = output["normal"]  # Normal in [h,w,3]
            moge_mask = output["mask"]
            depth = torch.where(moge_mask == 0, torch.tensor(1000.0, device=self.opts.device), depth)
            background_normal = torch.tensor([0.0, 0.0, -1.0], device=normal.device)
            normal = torch.where(moge_mask[..., None] == 0, background_normal[None, None, :], normal).view(-1, 3)
            masks = self.reliable_depth_mask_range_batch(depth.unsqueeze(0), window_size=5, ratio_thresh=0.1).view(-1).bool()
            K_normalized = output["intrinsics"]
            K = K_normalized.clone()
            K[0, 0] *= self.opts.width
            K[1, 1] *= self.opts.height
            K[0, 2] *= self.opts.width
            K[1, 2] *= self.opts.height
            depth = depth[None, None]
            K_inv = K.inverse()
            intrinsic = K[None].repeat(self.opts.video_length, 1, 1)
            return depth, masks, normal, intrinsic, K, K_inv

    def run_diffusion(self, cond_video, cond_masks, prompt, ref_video=None):
        cond_video = cond_video.permute(3, 0, 1, 2).unsqueeze(0)  # (1, T, H, W, C)
        cond_masks = cond_masks.permute(3, 0, 1, 2).unsqueeze(0)  # (1, T, H, W, C)
        if ref_video is not None:
            if ref_video.ndim == 4:
                ref_video = ref_video.permute(3, 0, 1, 2).unsqueeze(0)
            elif ref_video.ndim != 5:
                raise ValueError(f"ref_video must be 4D (T,H,W,C) or 5D (B,C,T,H,W), got {tuple(ref_video.shape)}")
        with self.aux_modules_offloaded():
            with torch.no_grad():
                steps = int(getattr(self.opts, "ddim_steps", 8))
                sample = self.pipeline(
                    video=cond_video,
                    mask=cond_masks,
                    ref_video=ref_video,
                    prompt=prompt,
                    negative_prompt=self.opts.negative_prompt,
                    height=self.opts.height,
                    width=self.opts.width,
                    num_frames=self.opts.video_length,
                    num_inference_steps=steps,
                    guidance_scale=float(getattr(self.opts, "diffusion_guidance_scale", 1.0)),
                    generator=torch.Generator(device=self.opts.device).manual_seed(42),
                ).frames[0]
        return torch.from_numpy(sample)

    def reliable_depth_mask_range_batch(self, depth, window_size=5, ratio_thresh=0.05, eps=1e-6):
        assert window_size % 2 == 1, "Window size must be odd."
        if depth.dim() == 3:
            depth_unsq = depth.unsqueeze(1)
        elif depth.dim() == 4:
            depth_unsq = depth
        else:
            raise ValueError("depth tensor must be of shape (B, H, W) or (B, 1, H, W)")
        local_max = torch.nn.functional.max_pool2d(depth_unsq, kernel_size=window_size, stride=1, padding=window_size // 2)
        local_min = -torch.nn.functional.max_pool2d(-depth_unsq, kernel_size=window_size, stride=1, padding=window_size // 2)
        local_mean = torch.nn.functional.avg_pool2d(depth_unsq, kernel_size=window_size, stride=1, padding=window_size // 2)
        ratio = (local_max - local_min) / (local_mean + eps)
        reliable_mask = (ratio < ratio_thresh) & (depth_unsq > 0)
        return reliable_mask

    def get_caption(self, image):
        inputs = self.caption_processor(images=image, return_tensors="pt").to(self.opts.device, torch.float16)
        generated_ids = self.captioner.generate(**inputs)
        generated_text = self.caption_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return generated_text + self.opts.refine_prompt

    def estimate_refer_video(self, frames_np):
        device = self.opts.device
        T, H, W, _ = frames_np.shape
        # STream3R inference: per-frame pose enc, depth, intrinsics
        imgs_stream = _preprocess_frames_for_stream3r(frames_np, mode="crop").to(device)  # (T,3,Hs,Ws)
        Hs, Ws = imgs_stream.shape[-2:]

        with torch.no_grad():
            # Process images one by one to simulate streaming inference
            outputs = {"pose_enc": [], "depth": []}
            for i in range(imgs_stream.shape[0]):
                image = imgs_stream[i : i + 1]
                outputs = self.session.forward_stream(image)
                
                if i==0: # Save KV cache of first frame
                    aggregator_kv_cache_list = self.session.aggregator_kv_cache_list
                    camera_head_kv_cache_list = self.session.camera_head_kv_cache_list
            
            self.session.clear()
            # Update KV cache for next inference
            self.session._update_cache(aggregator_kv_cache_list, camera_head_kv_cache_list)

        # pose encoding -> extrinsic/intrinsic at STream3R input resolution
        pose_enc = outputs["pose_enc"]  # (1,T,9)
        extri_34, intri_33 = pose_encoding_to_extri_intri(pose_enc, (Hs, Ws))  # (1,T,3,4), (1,T,3,3)
        extri_34 = extri_34.squeeze(0)  # (T,3,4)
        intri_33 = intri_33.squeeze(0)  # (T,3,3)

        # depth at STream3R input resolution -> resize back to original (H,W)
        depth_out = outputs["depth"]
        # Normalize to (B,S,1,Hs,Ws)
        if depth_out.ndim == 5 and depth_out.shape[-1] == 1:
            # (B,S,Hs,Ws,1) -> (B,S,1,Hs,Ws)
            depth_b_s_1_h_w = depth_out.permute(0, 1, 4, 2, 3).contiguous()
        elif depth_out.ndim == 5 and depth_out.shape[2] == 1:
            depth_b_s_1_h_w = depth_out
        else:
            raise ValueError(f"Unexpected depth tensor shape from STream3R: {tuple(depth_out.shape)}")

        depths_t = depth_b_s_1_h_w.squeeze(0).squeeze(1)  # (T,Hs,Ws)
        depths_t = F.interpolate(
            depths_t.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False
        ).squeeze(1)  # (T,H,W)

        # scale intrinsics to original resolution
        scale_x = float(W) / float(Ws)
        scale_y = float(H) / float(Hs)
        K_3x3 = intri_33.clone()
        K_3x3[:, 0, 0] *= scale_x
        K_3x3[:, 1, 1] *= scale_y
        K_3x3[:, 0, 2] *= scale_x
        K_3x3[:, 1, 2] *= scale_y
        K_inv = torch.linalg.inv(K_3x3)

        # Depth ready for downstream (no VDA alignment)
        # Convert to c2w (cam->world) and align to our world using first frame and initial camera
        Twcs_stream = closed_form_inverse_se3(extri_34)  # (T,4,4)

        return depths_t, Twcs_stream, K_3x3, K_inv



    def color_correct(self, diffusion_results: torch.Tensor, render_results: torch.Tensor, render_masks: torch.Tensor=None) -> torch.Tensor:
        """
        Color Correction: Use the first frame of render_results as reference to perform color mapping correction on all frames of diffusion_results

        Parameters:
            diffusion_results: (f, h, w, c) torch.Tensor, value range [0, 1]
            render_results:    (f, h, w, c) torch.Tensor, value range [0, 1]
            render_masks:      (f, h, w, c) torch.Tensor, value range [0, 1]

        Returns:
            corrected_results: torch.Tensor with same shape and dtype, color corrected
        """

        if render_masks is None:
            render_masks = torch.ones_like(render_results)

        #convert to numpy [0, 255]
        gen_first = (diffusion_results[0].cpu().numpy() * 255).astype(np.float32)
        ref_first = (render_results[0].cpu().numpy() * 255).astype(np.float32)
        mask_first = (render_masks[0].cpu().numpy()).astype(bool)

        # reshape to N x 3
        src_flat = gen_first.reshape(-1, 3)
        tgt_flat = ref_first.reshape(-1, 3)
        mask_flat = mask_first.reshape(-1)
        src_flat = src_flat[mask_flat]
        tgt_flat = tgt_flat[mask_flat]

        # color mapping: tgt = A * src + b
        X = np.hstack([src_flat, np.ones((src_flat.shape[0], 1))])
        Y = tgt_flat
        W, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        A = W[:3].T  # shape (3,3)
        b = W[3]     # shape (3,)

        # color correction
        corrected = []
        for i in range(diffusion_results.shape[0]):
            frame = (diffusion_results[i].cpu().numpy() * 255).astype(np.float32)
            h, w, c = frame.shape
            flat = frame.reshape(-1, 3)
            corrected_frame = flat @ A.T + b
            corrected_frame = np.clip(corrected_frame, 0, 255).reshape(h, w, 3).astype(np.uint8)
            corrected.append(corrected_frame)

        corrected_np = np.stack(corrected).astype(np.float32) / 255.0
        corrected_tensor = torch.from_numpy(corrected_np).to(diffusion_results.device).type(diffusion_results.dtype)
        return corrected_tensor

    def nvs_dynamic_view(self):
        with self.active_aux_modules():

            device = self.opts.device
            
            if bool(getattr(self.opts, "keep_aspect_ratio", False)):
                try:
                    from decord import VideoReader, cpu

                    vid = VideoReader(self.opts.image_dir, ctx=cpu(0))
                    first_frame = vid.get_batch([0]).asnumpy()
                    orig_h, orig_w = first_frame.shape[1:3]
                    target_h, target_w = self._compute_target_hw_keep_aspect_ratio(orig_h, orig_w)
                    self._set_resolution(target_h, target_w)
                except Exception as exc:
                    warnings.warn(f"Failed to infer input video aspect ratio, fallback to fixed size: {exc}")
                    self._set_resolution(int(getattr(self.opts, 'base_height', self.opts.height)), int(getattr(self.opts, 'base_width', self.opts.width)))
            else:
                self._set_resolution(int(getattr(self.opts, 'base_height', self.opts.height)), int(getattr(self.opts, 'base_width', self.opts.width)))
            
            # 0) init models    
            self.seg_net = TracerUniversalB7(device='cuda', batch_size=1, model_path=self.opts.segnet_path).eval()
            self.setup_stream3r()
            self.setup_sam2()
            
            # 1) Read video frames (numpy float32 in [0,1])
            frames_np = read_video_frames(
                self.opts.image_dir,
                self.opts.video_length,
                self.opts.stride,
                self.opts.max_res,
                width=int(self.opts.width),
                height=int(self.opts.height),
            )
            T, H, W, _ = frames_np.shape
            assert T == self.opts.video_length, "Wrong frame numbers"
            save_images_folder(frames_np, os.path.join(self.opts.save_dir, "images"))

            ###############
            # 2) Estimate depth using STream3R
            depths_t, Twcs_stream, K_3x3, K_inv = self.estimate_refer_video(frames_np)
            viz_depth_list([depths_t[i].cpu().numpy() for i in range(T)], self.opts.save_dir + "/depth_ref.mp4")

            ###############
            # 3) compute first-frame foreground center depth as radius and build new world frame
            image0 = (frames_np[0] * 255).astype(np.uint8)
            pil_image0 = Image.fromarray(image0)
            with torch.no_grad():
                origin_w_, origin_h_ = pil_image0.size
                image_pil = pil_image0.resize((512, 512))
                fg_mask0 = self.seg_net([image_pil])[0]
                fg_mask0 = fg_mask0.resize((origin_w_, origin_h_))
            fg_mask0 = np.array(fg_mask0) > 127.5
            fg_mask0 = torch.tensor(fg_mask0, device=device)
            if fg_mask0.float().mean() < 0.05:
                fg_mask0[...] = True
            depth0 = depths_t[0]
            depth_avg = torch.median(depth0[fg_mask0]).item()
            
            w2c_0, c2w_0 = set_initial_camera(self.opts.elevation, depth_avg)
            c2w_0 = c2w_0.to(device)
            # Align STream3R camera world to ours: Twc_new_i = c2w_0 @ inv(Twc_stream_0) @ Twc_stream_i
            Twc0_inv = torch.linalg.inv(Twcs_stream[0])
            Twn_ws = c2w_0 @ Twc0_inv
            Twcs_new = Twn_ws.unsqueeze(0) @ Twcs_stream  # (T,4,4)

            #####################
            # 4) Segment dynamic foreground using SAM2
            motion_fg_mask = fg_mask0
            # numpy opencv erode
            motion_fg_mask_ero = motion_fg_mask.cpu().numpy().astype(np.uint8)
            motion_fg_mask_ero = cv2.erode(motion_fg_mask_ero, np.ones((9, 9), np.uint8), iterations=2)
            motion_fg_mask_ero = torch.from_numpy(motion_fg_mask_ero).to(device).type(torch.float32)

            sample_interval = 100
            sample_mask = np.zeros((1, H, W), dtype=np.uint8)
            sample_mask[:, ::sample_interval, ::sample_interval] = 1
            sample_mask = torch.from_numpy(sample_mask).to(device)
            sample_motion_fg_mask = motion_fg_mask_ero * sample_mask
            # Adaptively sample prompt points based on mask size
            while sample_interval > 10:
                if sample_motion_fg_mask.sum() < 10:
                    sample_interval = sample_interval - 10
                    sample_mask = np.zeros((1, H, W), dtype=np.uint8)
                    sample_mask[:, ::sample_interval, ::sample_interval] = 1
                    sample_mask = torch.from_numpy(sample_mask).to(device)
                    sample_motion_fg_mask = motion_fg_mask_ero * sample_mask
                else:
                    break
            print('Sample points num, ', sample_motion_fg_mask.sum())
            mask_idx = torch.nonzero(sample_motion_fg_mask[0], as_tuple=False)  # → (N, 2) LongTensor
            sam_points = mask_idx[:, [1, 0]].int().cpu().numpy()
            sam_labels = np.ones(sam_points.shape[0], dtype=np.int32)
            self.sam_points = sam_points
            
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):            
                state = self.sampredictor.init_state(os.path.join(self.opts.save_dir, "images"))  # init sam2
                # Segment point by point
                for label_i in range(sam_labels.shape[0]): 
                    sam_masks_cur, sam_painted_images = self.run_sam2_video((frames_np * 255).astype(np.uint8), state, sam_points[label_i:label_i+1], sam_labels[label_i:label_i+1], None)
                    self.sampredictor.reset_state(state)
                    if label_i == 0:
                        sam_masks = sam_masks_cur
                    else:
                        for t in range(len(sam_masks_cur)):
                            sam_masks[t] = (sam_masks[t] + sam_masks_cur[t]) > 0 # merge
            # background
            sam_masks = 1 - torch.from_numpy(np.stack(sam_masks, axis=0)[...,None]).float().permute(0, 3, 1, 2).to(device) #[T, 1, H, W]
            inverse_sam_masks = (1-sam_masks).cpu().numpy()
            for t in range(inverse_sam_masks.shape[0]): # eliminate cracks in foreground mask
                inverse_sam_masks[t,0] = cv2.dilate(inverse_sam_masks[t,0], np.ones((3, 3), np.uint8), iterations=4)
                inverse_sam_masks[t,0] = cv2.erode(inverse_sam_masks[t,0], np.ones((3, 3), np.uint8), iterations=4)
            sam_masks = 1 - torch.from_numpy(inverse_sam_masks).float().to(device) # background
            painted_images = torch.from_numpy(np.stack(sam_painted_images, axis=0))
            os.makedirs(os.path.join(self.opts.save_dir, 'sam'), exist_ok=True)

            save_video(
                (1-sam_masks.permute(0, 2, 3, 1).repeat(1, 1, 1, 3)),
                os.path.join(self.opts.save_dir, 'sam', 'masks.mp4'),
                fps=self.opts.fps,
            )
            save_video(
                painted_images / 255.0,
                os.path.join(self.opts.save_dir, 'sam', 'painted_images.mp4'),
                fps=self.opts.fps,
            )
            np.save(os.path.join(self.opts.save_dir, 'sam', 'masks.npy'), sam_masks.cpu().numpy())

            torch.cuda.empty_cache()
            gc.collect()

            #####################
            # 5) Inpaint background using ROSE
            frames_torch = torch.from_numpy(frames_np).to(device).to(torch.float32)  # (T,H,W,3) in [0,1]
            view_masks = (1-sam_masks.permute(0, 2, 3, 1)).to(device)
            target_h, target_w = int(self.opts.height), int(self.opts.width)
            diffusion_results1 = video_inpainting(self.opts.rose_ckpt, self.opts.rose_model, frames_torch, view_masks, "", T, [target_h, target_w])
            diffusion_results1 = F.interpolate(diffusion_results1, size=[target_h, target_w], mode='bilinear', align_corners=False).permute(0,2,3,1)
            diffusion_results = self.color_correct(diffusion_results1, frames_torch, 1-view_masks)
            save_video(diffusion_results1, os.path.join(self.opts.save_dir, 'background_uncorrect.mp4'))
            save_video(diffusion_results.cpu(), os.path.join(self.opts.save_dir, 'background_correct.mp4'))

            #####################
            # 6) Estimate Depth of inpainted background and align foreground depth using vda
            frames_np_bg = diffusion_results.cpu().numpy()
            depth_bg, Twcs_stream, K_3x3, K_inv = self.estimate_refer_video(frames_np_bg)
            viz_depth_list([depth_bg[i].cpu().numpy() for i in range(T)], self.opts.save_dir + "/depth_bg.mp4")
            dep_list = [depth_cur for depth_cur in depth_bg.cpu().numpy()]
            fn_list = [f"{i:05d}.png" for i in range(T)]
            save_depth_list(dep_list, fn_list, os.path.join(self.opts.save_dir, 'stream3r_depth_bg'), None)

            frames_list_float = [(frames_np[i]) for i in range(T)]
            aligned_depths_t = self.align_video_depth(frames_list_float, depths_t, 1-sam_masks.squeeze())
            depth_fg = (1-sam_masks.squeeze()) * aligned_depths_t
            viz_depth_list([depth_fg[i].cpu().numpy() for i in range(T)], self.opts.save_dir + "/depth_fg.mp4")
            
            depths_t = sam_masks.squeeze() * depth_bg + (1-sam_masks.squeeze()) * depth_fg
            depth_masks = self.reliable_depth_mask_range_batch(depths_t.unsqueeze(1), window_size=5, ratio_thresh=0.1).squeeze()  # (T,1,H,W)
            max_depth = torch.max(depths_t)
            depths_t_viz = depth_masks * depths_t + ~depth_masks*max_depth
            viz_depth_list([depths_t_viz[i].cpu().numpy() for i in range(T)], self.opts.save_dir + "/depth_align.mp4")
            dep_list = [depth_cur for depth_cur in depths_t.cpu().numpy()]
            save_depth_list(dep_list, fn_list, os.path.join(self.opts.save_dir, 'stream3r_depth'), None)
            
            depth0 = depths_t[0]
            depth_avg = torch.median(depth0[fg_mask0])
            
            #####################
            # 7) Save data for reconstruction
            json_data = {
                "H": H,
                "W": W,
                "pose": Twcs_new.cpu().numpy().tolist(), # c2w
                "K": K_3x3.cpu().numpy().tolist(),
                "depth_avg": depth_avg.item(), # avg depth of foreground
                "sam_points": self.sam_points.tolist(), # prompt points for sam2
            }
            json.dump(json_data, open(os.path.join(self.opts.save_dir, "poses.json"), "w"))


    def align_bg_depth(self):
        T = self.opts.video_length
        device = self.device

        fn_list = [f"{i:05d}.png" for i in range(T)]
        frame_names = [f"{i:05d}" for i in range(T)]
        
        sam_masks = np.load(os.path.join(self.opts.save_dir, 'sam', 'masks.npy'))
        sam_masks = torch.from_numpy(sam_masks).to(device)

        # bundle results from MoSca
        bundle_pth_fn = os.path.join(self.opts.save_dir, "bundle", "bundle.pth")
        bundle_data = torch.load(bundle_pth_fn)
        dep_scale = bundle_data["dep_scale"]

        depths_t = []
        depth_bg = []

        for index, img_name in enumerate(frame_names):
            # fg depth
            dep_file = os.path.join(self.opts.save_dir, 'stream3r_depth', f"{img_name}.npz")
            dep_raw = np.load(dep_file)["dep"]
            scale = dep_scale[index].detach().cpu().numpy()
            depths_t.append(dep_raw * (1-sam_masks[index].squeeze().cpu().numpy()) + dep_raw * scale * sam_masks[index].squeeze().cpu().numpy())

            # bg depth
            dep_file_bg = os.path.join(self.opts.save_dir, 'stream3r_depth_bg', f"{img_name}.npz")
            dep_raw_bg = np.load(dep_file_bg)["dep"]
            depth_bg.append(dep_raw_bg * scale)

        save_depth_list(depths_t, fn_list, os.path.join(self.opts.save_dir, 'stream3r_depth'), None)
        save_depth_list(depth_bg, fn_list, os.path.join(self.opts.save_dir, 'stream3r_depth_bg'), None)


    def gen_sup_views(self, dir_name):
        device = self.device
        
        if bool(getattr(self.opts, "keep_aspect_ratio", False)):
            try:
                from decord import VideoReader, cpu

                vid = VideoReader(self.opts.image_dir, ctx=cpu(0))
                first_frame = vid.get_batch([0]).asnumpy()
                orig_h, orig_w = first_frame.shape[1:3]
                target_h, target_w = self._compute_target_hw_keep_aspect_ratio(orig_h, orig_w)
                self._set_resolution(target_h, target_w)
            except Exception as exc:
                warnings.warn(f"Failed to infer input video aspect ratio, fallback to fixed size: {exc}")
                self._set_resolution(int(getattr(self.opts, 'base_height', self.opts.height)), int(getattr(self.opts, 'base_width', self.opts.width)))
        else:
            self._set_resolution(int(getattr(self.opts, 'base_height', self.opts.height)), int(getattr(self.opts, 'base_width', self.opts.width)))


        #####################
        # 0) Load data
        frames_np = read_video_frames(
            self.opts.image_dir,
            self.opts.video_length,
            self.opts.stride,
            self.opts.max_res,
            width=int(self.opts.width),
            height=int(self.opts.height),
        )
        T, H, W, _ = frames_np.shape
        assert T == self.opts.video_length, "Wrong frame numbers"

        ## load json
        json_path = os.path.join(self.opts.save_dir, "poses.json")
        with open(json_path, 'r', encoding='utf-8') as json_file:
            json_data = json.load(json_file)
            depth_avg = json_data["depth_avg"]
            self.sam_points = np.array(json_data["sam_points"])
        sam_labels = np.ones(self.sam_points.shape[0], dtype=np.int32)
        sam_masks = np.load(os.path.join(self.opts.save_dir, 'sam', 'masks.npy'))
        sam_masks = torch.from_numpy(sam_masks).to(device)

        frames_np_bg = read_video_frames(
            os.path.join(self.opts.save_dir, 'background_correct.mp4'),
            self.opts.video_length,
            self.opts.stride,
            self.opts.max_res,
            width=int(self.opts.width),
            height=int(self.opts.height),
        )
        fn_list = [f"{i:05d}.png" for i in range(T)]
        frame_names = [f"{i:05d}" for i in range(T)]

        depths_t = []
        depth_bg = []

        for index, img_name in enumerate(frame_names):
            dep_file = os.path.join(self.opts.save_dir, 'stream3r_depth', f"{img_name}.npz")
            dep_raw = np.load(dep_file)["dep"]
            depths_t.append(dep_raw)
            dep_file_bg = os.path.join(self.opts.save_dir, 'stream3r_depth_bg', f"{img_name}.npz")
            dep_raw_bg = np.load(dep_file_bg)["dep"]
            depth_bg.append(dep_raw_bg)

        depths_t = torch.from_numpy(np.stack(depths_t)).to(device)
        depth_bg = torch.from_numpy(np.stack(depth_bg)).to(device)
        depths_nofg = sam_masks.squeeze() * depth_bg + (1-sam_masks.squeeze()) * depth_avg

        depth_masks = self.reliable_depth_mask_range_batch(depths_nofg.unsqueeze(1), window_size=5, ratio_thresh=0.1)  # (T,1,H,W)
        frames_torch = torch.from_numpy(frames_np).to(device).to(torch.float32)  # (T,H,W,3) in [0,1]
        depth_masks_bg = self.reliable_depth_mask_range_batch(depth_bg.unsqueeze(1), window_size=5, ratio_thresh=0.1)
        frames_torch_bg = torch.from_numpy(frames_np_bg).to(device).to(torch.float32)
        
        ckpt = torch.load(os.path.join(self.opts.save_dir, "bundle", "bundle_cams.pth"))
        H = ckpt["default_H"]
        W = ckpt["default_W"]
        delta_flag = ckpt["delta_flag"]

        if "iso_focal" not in ckpt.keys():
            ckpt["iso_focal"] = torch.tensor(False)
        if "_rel_focal" not in ckpt.keys():
            ckpt["_rel_focal"] = ckpt["rel_focal"]
            del ckpt["rel_focal"]
        T = len(ckpt["q_wc"])
        rel_focal = ckpt["_rel_focal"]
        if ckpt["iso_focal"]:
            rel_focal = rel_focal[0].repeat(2)
        cxcy_ratio = ckpt["cxcy_ratio"]
        q_wc = ckpt["q_wc"]
        t_wc = ckpt["t_wc"]
        
        Twcs_new = torch.zeros(T, 4, 4).to(device)
        for ind in range(T):
            R = quaternion_to_matrix(F.normalize(q_wc[ind : ind + 1], dim=-1))[0]
            t = t_wc[ind]
            #T = torch.eye(4).to(R)
            Twcs_new[ind, :3, :3] = R
            Twcs_new[ind, :3, 3] = t
            Twcs_new[ind, 3, 3] = 1


        #####################
        # 1) Estimate Noraml using MoGe
        if H is None and W is None:
            return default_K
        else:
            assert H is not None and W is not None, "H and W must be both provided"
        L = min(H, W)  # ! the rel means to rel to the short side
        fx = rel_focal[0] * L / 2.0
        fy = rel_focal[1] * L / 2.0
        cx = W * cxcy_ratio[0]
        cy = H * cxcy_ratio[1]
        K = torch.eye(3).to(rel_focal)
        K[0, 0] = K[0, 0] * 0 + fx
        K[1, 1] = K[1, 1] * 0 + fy
        K[0, 2] = K[0, 2] * 0 + cx
        K[1, 2] = K[1, 2] * 0 + cy
        K_3x3 = K.unsqueeze(0).repeat(T, 1, 1)
        K_inv = torch.linalg.inv(K_3x3)

        depths_nofg = sam_masks.squeeze() * depth_bg + (1-sam_masks.squeeze()) * depth_avg
        # Compute MoGe normals per frame (batched) and rotate to world
        normals_cam_list = []
        with torch.no_grad():
            bs = 9
            frames_tensor = torch.from_numpy(frames_np).to(device).to(torch.float32)
            for start in range(0, T, bs):
                end = min(start + bs, T)
                for j in range(start, end):
                    img_chw = frames_tensor[j].permute(2, 0, 1)
                    out = self.depth_model.infer(img_chw)
                    normal = out["normal"]
                    moge_mask = out["mask"]
                    background_normal = torch.tensor([0.0, 0.0, -1.0], device=normal.device, dtype=normal.dtype)
                    normal = torch.where(moge_mask[..., None] == 0, background_normal[None, None, :], normal)
                    normals_cam_list.append(normal)

        normals_world_list = []
        for i in range(T):
            R_c2w = Twcs_new[i][:3, :3]
            n_cam = normals_cam_list[i].reshape(-1, 3)
            n_world = (R_c2w @ n_cam.T).T
            n_world = F.normalize(n_world, dim=-1, eps=1e-6)
            normals_world_list.append(n_world.view(H, W, 3))

        normals_world = torch.stack(normals_world_list, dim=0)  # (T,H,W,3)

        #####################
        # 2) Unproject each frame to world points via our own math (consistent with existing pipeline)
        uu, vv = torch.meshgrid(
            torch.arange(W, device=device, dtype=torch.float32),
            torch.arange(H, device=device, dtype=torch.float32),
            indexing="xy",
        )
        uv1 = torch.stack([uu, vv, torch.ones_like(uu)], dim=-1).view(1, H * W, 3).repeat(T, 1, 1)  # (T,HW,3)
        depths_hw = depths_t.view(T, H * W)
        depths_hw_bg = depth_bg.view(T, H * W)

        points_world_list = []
        colors_list = []
        mask_list = []
        points_world_list_bg = []
        colors_list_bg = []
        mask_list_bg = []

        depth_masks = self.reliable_depth_mask_range_batch(depths_nofg.unsqueeze(1), window_size=5, ratio_thresh=0.1)  # (T,1,H,W)
        frames_torch = torch.from_numpy(frames_np).to(device).to(torch.float32)  # (T,H,W,3) in [0,1]
        depth_masks_bg = self.reliable_depth_mask_range_batch(depth_bg.unsqueeze(1), window_size=5, ratio_thresh=0.1)
        frames_torch_bg = torch.from_numpy(frames_np_bg).to(device).to(torch.float32)

        for i in range(T):
            Ki_inv = K_inv[i]
            rays = (Ki_inv @ uv1[i].T).T               # (HW,3)
            Xi_cam = rays * depths_hw[i][:, None]      # (HW,3)
            Xi_cam_bg = rays * depths_hw_bg[i][:, None]

            R_c2w = Twcs_new[i][:3, :3]
            t_c2w = Twcs_new[i][:3, 3]
            Xi_world = (R_c2w @ Xi_cam.T).T + t_c2w[None]
            Xi_world = Xi_world.view(H, W, 3)
            Xi_world_bg = (R_c2w @ Xi_cam_bg.T).T + t_c2w[None]
            Xi_world_bg = Xi_world_bg.view(H, W, 3)

            points_world_list.append(Xi_world)
            colors_list.append(frames_torch[i] * 2.0 - 1.0)
            mask_list.append(depth_masks[i])
            points_world_list_bg.append(Xi_world_bg)
            colors_list_bg.append(frames_torch_bg[i] * 2.0 - 1.0)
            mask_list_bg.append(depth_masks_bg[i])

        points_world = torch.stack(points_world_list, dim=0)  # (T,H,W,3)
        colors = torch.stack(colors_list, dim=0)              # (T,H,W,3)
        masks = torch.stack(mask_list, dim=0).float()         # (T,1,H,W)
        points_world_bg = torch.stack(points_world_list_bg, dim=0)
        colors_bg = torch.stack(colors_list_bg, dim=0)
        masks_bg = torch.stack(mask_list_bg, dim=0).float()

        self.opts.save_dir = os.path.join(self.opts.save_dir, dir_name)
        os.makedirs(self.opts.save_dir, exist_ok=True)

        #####################
        # 3) Build target camera trajectory and intrinsics (reuse existing util)
        if self.opts.traj_type == "custom" or self.opts.traj_type == "freeze":
            cam_traj, x_offset, y_offset, z_offset, d_theta, d_phi, d_r = (
                "free",
                self.opts.x_offset,
                self.opts.y_offset,
                self.opts.z_offset,
                self.opts.d_theta,
                self.opts.d_phi,
                self.opts.d_r,
            )
        elif self.opts.traj_type == "rel_target":
            cam_traj, x_offset, y_offset, z_offset, d_theta, d_phi, d_r = (
                self.opts.traj_type,
                self.opts.x_offset,
                self.opts.y_offset,
                self.opts.z_offset,
                self.opts.d_theta,
                self.opts.d_phi,
                self.opts.d_r,
            )
        else:
            cam_traj, x_offset, y_offset, z_offset, d_theta, d_phi, d_r = traj_map(self.opts.traj_type)

        intrinsic_base_cpu = K_3x3[0].detach().cpu()
        c2w_0 = Twcs_new[0]
        c2w_0_cpu = c2w_0.detach().cpu()
        w2c_0_cpu = torch.linalg.inv(c2w_0_cpu)

        w2cs_target, c2ws_target, intrinsic_target = build_cameras(
            cam_traj="free",
            w2c_0=w2c_0_cpu,
            c2w_0=c2w_0_cpu,
            intrinsic=intrinsic_base_cpu,
            nframe=T,
            focal_length=self.opts.focal_length,
            d_theta=d_theta,
            d_phi=d_phi,
            d_r=d_r,
            radius=depth_avg * getattr(self.opts, "radius_scale", 1.0),
            x_offset=x_offset,
            y_offset=y_offset,
            z_offset=z_offset,
        )
        c2ws_target = c2ws_target.to(device)
        intrinsic_target = intrinsic_target.to(device)
        # reverse
        w2cs_target_reverse, c2ws_target_reverse, _ = build_cameras(
            cam_traj="free",
            w2c_0=w2c_0_cpu,
            c2w_0=c2w_0_cpu,
            intrinsic=intrinsic_base_cpu,
            nframe=T,
            focal_length=self.opts.focal_length,
            d_theta=-d_theta,
            d_phi=-d_phi,
            d_r=d_r,
            radius=depth_avg * getattr(self.opts, "radius_scale", 1.0),
            x_offset=-x_offset,
            y_offset=-y_offset,
            z_offset=-z_offset,
        )
        c2ws_target_reverse = c2ws_target_reverse.to(device)


        #####################
        # 4) Static view augmentation (rotate first frame)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            # Occlusion-aware Point Cloud Rendering
            points_world_input = points_world.clone()
            colors_input = colors.clone()
            normals_world_input = normals_world.clone()
            masks_input = masks.clone()
            for i in range(T):
                points_world_input[i] = points_world[0]
                colors_input[i] = colors[0]
                normals_world_input[i] = normals_world[0]
                masks_input[i] = (1-sam_masks[0]) * masks[0]
            control_imgs_all, inconsis_mask, _, _ = self.renderer.render(
                c2ws=c2ws_target,
                Ks=intrinsic_target,
                points_world_from_img1=points_world_input,
                colors_from_img1=colors_input,
                mask_img1=masks_input,
                normal_world_from_img1=normals_world_input,
                vis_threshold=getattr(self.opts, "vis_threshold", -0.1),
            )
            for i in range(T):
                points_world_input[i] = points_world_bg[0]
                colors_input[i] = colors_bg[0]
                #normals_world_input[i] = normals_world_bg[0]
                masks_input[i] = masks_bg[0]
            control_imgs_bg, render_masks_bg, render_depth_bg, render_depth_masks_bg = self.renderer.render(
                c2ws=c2ws_target,
                Ks=intrinsic_target,
                points_world_from_img1=points_world_input,
                colors_from_img1=colors_input,
                mask_img1=masks_input,
                normal_world_from_img1=None,
                vis_threshold=getattr(self.opts, "vis_threshold", -0.1),
            )
            save_video(inconsis_mask.permute(0,2,3,1).repeat(1, 1, 1, 3), os.path.join(self.opts.save_dir, 'mask_inconsis.mp4'), fps=self.opts.fps)

            # reverse warp
            depths_hw = render_depth_bg.view(T, H * W)
            for i in range(T):
                Ki_inv = K_inv[0]
                rays = (Ki_inv @ uv1[i].T).T               # (HW,3)
                Xi_cam = rays * depths_hw[i][:, None]      # (HW,3)

                R_c2w = Twcs_new[0][:3, :3]
                t_c2w = Twcs_new[0][:3, 3]
                Xi_world = (R_c2w @ Xi_cam.T).T + t_c2w[None]
                Xi_world = Xi_world.view(H, W, 3)

                points_world_input[i] = Xi_world
                colors_input[i] = control_imgs_bg[i].permute(1, 2, 0)
                masks_input[i] = 1-inconsis_mask[i]

            # obtain occlusion relationship
            _, _, _, render_masks_occ = self.renderer.render(
                c2ws=c2ws_target_reverse,
                Ks=intrinsic_target,
                points_world_from_img1=points_world_input,
                colors_from_img1=colors_input,
                mask_img1=masks_input,
                normal_world_from_img1=None,
                vis_threshold=getattr(self.opts, "vis_threshold", -0.1),
            )
            save_video(render_masks_occ.permute(0,2,3,1).repeat(1, 1, 1, 3), os.path.join(self.opts.save_dir, 'mask_occ.mp4'), fps=self.opts.fps)

            for i in range(T):
                if i == 0:
                    masks_input[0] = render_masks_occ[0]
                else:
                    masks_input[i] = masks_input[i-1] * render_masks_occ[i]
                masks_input[i] = masks_input[i] + (1-sam_masks[0])
                points_world_input[i] = points_world[0]
                colors_input[i] = colors[0]
            masks_input[masks_input>0] = 1
            masks_input *= masks[0:1]
            control_imgs, render_masks, _, _ = self.renderer.render(
                c2ws=c2ws_target,
                Ks=intrinsic_target,
                points_world_from_img1=points_world_input,
                colors_from_img1=colors_input,
                mask_img1=masks_input,
                normal_world_from_img1=normals_world_input,
                vis_threshold=getattr(self.opts, "vis_threshold", -0.1),
            )
            save_video(render_masks.permute(0,2,3,1).repeat(1, 1, 1, 3), os.path.join(self.opts.save_dir, 'mask_1001.mp4'), fps=self.opts.fps)

        target_h, target_w = int(self.opts.height), int(self.opts.width)
        control_imgs = F.interpolate(control_imgs, size=[target_h, target_w], mode='bilinear', align_corners=False)
        render_masks = F.interpolate(render_masks.float(), size=[target_h, target_w], mode='nearest')
        control_imgs[0:1] = (frames_torch[0:1]).permute(0, 3, 1, 2) * 2.0 - 1.0
        render_masks[0:1] = 1.0

        frames_vis = frames_torch
        save_video(frames_vis, os.path.join(self.opts.save_dir, 'input.mp4'), fps=self.opts.fps)

        render_results = einops.rearrange(control_imgs, "f c h w -> f h w c", f=T)
        view_masks = einops.rearrange(render_masks, "f c h w -> f h w c", f=T)
        view_masks = 1.0 - view_masks

        save_video((render_results + 1.0) / 2.0, os.path.join(self.opts.save_dir, 'render.mp4'), fps=self.opts.fps)
        save_video(view_masks.repeat(1, 1, 1, 3), os.path.join(self.opts.save_dir, 'mask.mp4'), fps=self.opts.fps)

        # generation
        if not hasattr(self, 'pipeline'):
            self.setup_diffusion()
        pil_mid = Image.fromarray((frames_np[T // 2] * 255).astype(np.uint8))
        prompt = self.get_caption(pil_mid)
        print(prompt)
        ref_video = (frames_torch[:10].permute(3, 0, 1, 2).unsqueeze(0) * 2.0 - 1.0)
        diffusion_results1 = self.run_diffusion(render_results, view_masks, prompt, ref_video=ref_video)
        diffusion_results = self.color_correct(diffusion_results1, (render_results + 1.0) / 2.0, 1-view_masks)
        save_video(diffusion_results1, os.path.join(self.opts.save_dir, 'diffusion_uncorrect.mp4'))
        save_video(diffusion_results, os.path.join(self.opts.save_dir, 'diffusion_correct.mp4'))
        tensor_left = frames_vis.to(device)
        tensor_right = diffusion_results.to(device)
        diffusion_results = diffusion_results.to(device)
        interval = torch.ones(T, self.opts.height, 30, 3, device=device)
        final_result = torch.cat((tensor_left, interval, tensor_right), dim=2)
        save_video(final_result, os.path.join(self.opts.save_dir, 'diffusion.mp4'), fps=self.opts.fps)
        
        # Segment foreground using sam2
        frames_list = [(diffusion_results[i] * 255).cpu().numpy().astype(np.uint8) for i in range(T)]
        template_mask = 1 - sam_masks[0, 0].cpu().numpy()
        if not hasattr(self, 'sampredictor'):
            self.setup_sam2()
        sam_labels = np.ones(self.sam_points.shape[0], dtype=np.int32)
        save_images_folder(diffusion_results.cpu().numpy(), os.path.join(self.opts.save_dir, "tmp_dir"))
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):            
            state = self.sampredictor.init_state(os.path.join(self.opts.save_dir, "tmp_dir"))
            for label_i in range(sam_labels.shape[0]):
                sam_masks_cur, sam_painted_images = self.run_sam2_video((diffusion_results.cpu().numpy() * 255).astype(np.uint8), state, self.sam_points[label_i:label_i+1], sam_labels[label_i:label_i+1], None)
                self.sampredictor.reset_state(state)
                if label_i == 0:
                    sam_masks_rotate = sam_masks_cur
                else:
                    for t in range(len(sam_masks_cur)):
                        sam_masks_rotate[t] = (sam_masks_rotate[t] + sam_masks_cur[t]) > 0
        
        sam_masks_rotate = 1 - torch.from_numpy(np.stack(sam_masks_rotate, axis=0)[...,None]).float().permute(0, 3, 1, 2) #[T, 1, H, W]
        sam_masks_rotate = sam_masks_rotate.to(device)
        painted_images = torch.from_numpy(np.stack(sam_painted_images, axis=0))
        os.makedirs(os.path.join(self.opts.save_dir, 'sam_rotate'), exist_ok=True)
        
        save_video(
            (1-sam_masks_rotate.permute(0, 2, 3, 1).repeat(1, 1, 1, 3)),
            os.path.join(self.opts.save_dir, 'sam_rotate', 'masks.mp4'),
            fps=self.opts.fps,
        )
        save_video(
            painted_images / 255.0,
            os.path.join(self.opts.save_dir, 'sam_rotate', 'painted_images.mp4'),
            fps=self.opts.fps,
        )
        torch.cuda.empty_cache()
        gc.collect()


        #####################
        # 4) Dynamic view augmentation iteration
        iter_num = 4
        c2w_0_cpu = c2w_0.detach().cpu()
        w2c_0_cpu = torch.linalg.inv(c2w_0_cpu)
        w2cs_target, c2ws_target, intrinsic_target = build_cameras(
            cam_traj='rel_target',
            w2c_0=Twcs_new.cpu(),
            c2w_0=Twcs_new.cpu(),
            intrinsic=K_3x3.detach().cpu(),
            nframe=T,
            focal_length=self.opts.focal_length,
            d_theta=d_theta / iter_num,
            d_phi=d_phi / iter_num,
            d_r=d_r,
            radius=depth_avg * getattr(self.opts, "radius_scale", 1.0),
            x_offset=x_offset,
            y_offset=y_offset,
            z_offset=z_offset,
        )
        c2ws_target = c2ws_target.to(device)
        intrinsic_target = intrinsic_target.to(device)

        points_world_pre = points_world.clone()
        colors_pre = colors.clone()
        masks_pre = masks.clone()

        for iteri in range(iter_num):
            # rotation angle
            d_phi_cur = d_phi / iter_num * (iteri+1)
            d_theta_cur = d_theta / iter_num * (iteri+1)

            _, c2ws_target_cur, intrinsic_target_cur = build_cameras(
                cam_traj='rel_target',
                w2c_0=Twcs_new.cpu(),
                c2w_0=Twcs_new.cpu(),
                intrinsic=K_3x3.detach().cpu(),
                nframe=T,
                focal_length=self.opts.focal_length,
                d_theta=d_theta_cur,
                d_phi=d_phi_cur,
                d_r=d_r,
                radius=depth_avg * getattr(self.opts, "radius_scale", 1.0),
                x_offset=x_offset,
                y_offset=y_offset,
                z_offset=z_offset,
            )
            c2ws_target_cur = c2ws_target_cur.to(device)
            intrinsic_target_cur = intrinsic_target_cur.to(device)
            
            _, c2ws_target_cur_reverse, _ = build_cameras(
                cam_traj='rel_target',
                w2c_0=Twcs_new.cpu(),
                c2w_0=Twcs_new.cpu(),
                intrinsic=K_3x3.detach().cpu(),
                nframe=T,
                focal_length=self.opts.focal_length,
                d_theta=-d_theta_cur,
                d_phi=-d_phi_cur,
                d_r=d_r,
                radius=depth_avg * getattr(self.opts, "radius_scale", 1.0),
                x_offset=-x_offset,
                y_offset=-y_offset,
                z_offset=-z_offset,
            )
            c2ws_target_cur_reverse = c2ws_target_cur_reverse.to(device)
            print(H, W)
            json_data = {
                "H": H.item(),
                "W": W.item(),
                "pose": c2ws_target_cur.cpu().numpy().tolist(),
                "K": intrinsic_target_cur.cpu().numpy().tolist(),
            }
            json.dump(json_data, open(os.path.join(self.opts.save_dir, f"poses_sup{iteri}.json"), "w"))
            
            inverse_sam_masks_rotate = 1-sam_masks_rotate
            inverse_sam_masks_rotate = inverse_sam_masks_rotate.cpu().numpy()
            for t in range(inverse_sam_masks_rotate.shape[0]):
                inverse_sam_masks_rotate[t, 0] = cv2.dilate(inverse_sam_masks_rotate[t, 0], np.ones((5, 5), np.uint8), iterations=2)
                inverse_sam_masks_rotate[t, 0] = cv2.erode(inverse_sam_masks_rotate[t, 0], np.ones((5, 5), np.uint8), iterations=2)
            sam_masks_rotate = 1 - torch.from_numpy(inverse_sam_masks_rotate).float().to(device)

            with self.aux_modules_offloaded():
                with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
                    
                    if iteri == 0:
                        occlusion_masks = 1-sam_masks
                    occlusion_masks[0:1] = 0.0
                    
                    _, _, render_depth_bg_full, render_depth_masks_bg_full = self.renderer.render(
                        c2ws=c2ws_target_cur,
                        Ks=intrinsic_target_cur,
                        points_world_from_img1=points_world_bg,
                        colors_from_img1=colors_bg,
                        mask_img1=masks_bg,
                        normal_world_from_img1=None,
                        vis_threshold=getattr(self.opts, "vis_threshold", -0.1),
                    )
                    control_imgs_bg, render_masks_bg, render_depth_bg, _ = self.renderer.render(
                        c2ws=c2ws_target_cur,
                        Ks=intrinsic_target_cur,
                        points_world_from_img1=points_world_bg,
                        colors_from_img1=colors_bg,
                        mask_img1=masks_bg * (1-occlusion_masks),
                        normal_world_from_img1=None,
                        vis_threshold=getattr(self.opts, "vis_threshold", -0.1),
                    )
                    control_imgs_fg, render_masks_fg, _, _ = self.renderer.render(
                        c2ws=c2ws_target_cur,
                        Ks=intrinsic_target_cur,
                        points_world_from_img1=points_world,
                        colors_from_img1=colors,
                        mask_img1=masks *(1-sam_masks),
                        normal_world_from_img1=normals_world,
                        vis_threshold=getattr(self.opts, "vis_threshold", -0.1),
                    )
                    torch.cuda.empty_cache()
                    gc.collect()

            control_imgs_bg = F.interpolate(control_imgs_bg, size=[target_h, target_w], mode='bilinear', align_corners=False)
            render_masks_bg = F.interpolate(render_masks_bg.float(), size=[target_h, target_w], mode='nearest')
            control_imgs_fg = F.interpolate(control_imgs_fg, size=[target_h, target_w], mode='bilinear', align_corners=False)
            render_masks_fg = F.interpolate(render_masks_fg.float(), size=[target_h, target_w], mode='nearest')

            rotate_T = (T-1)// iter_num * (iteri+1)
            control_imgs = (control_imgs_fg + 1.0) / 2.0 * render_masks_fg + (control_imgs_bg + 1.0) / 2.0 * (1-render_masks_fg)*render_masks_bg
            control_imgs[0:1] = (diffusion_results[rotate_T:rotate_T+1]).permute(0, 3, 1, 2) * (1-sam_masks_rotate[rotate_T:rotate_T+1]) + \
                                ((control_imgs_bg[0:1] + 1.0) / 2.0 * sam_masks_rotate[rotate_T:rotate_T+1]) 
            control_imgs = control_imgs * 2.0 - 1.0
            render_masks = render_masks_bg + render_masks_fg
            render_masks[0:1] = render_masks_bg[0:1] + (1-sam_masks_rotate[rotate_T:rotate_T+1])
            render_masks[render_masks > 0] = 1.0
            frames_vis = frames_torch

            render_results = einops.rearrange(control_imgs, "f c h w -> f h w c", f=T)
            view_masks = einops.rearrange(render_masks, "f c h w -> f h w c", f=T)
            view_masks = 1.0 - view_masks

            save_video((render_results + 1.0) / 2.0, os.path.join(self.opts.save_dir, f'render_sup{iteri}.mp4'), fps=self.opts.fps)
            save_video(view_masks.repeat(1, 1, 1, 3), os.path.join(self.opts.save_dir, f'mask_sup{iteri}.mp4'), fps=self.opts.fps)

            # generation
            print(prompt)
            ref_video = (frames_torch[:10].permute(3, 0, 1, 2).unsqueeze(0) * 2.0 - 1.0)
            diffusion_results1 = self.run_diffusion(render_results, view_masks, prompt, ref_video=ref_video)
            diffusion_results_cur = self.color_correct(diffusion_results1, (render_results + 1.0) / 2.0, 1-view_masks)
            save_video(diffusion_results1, os.path.join(self.opts.save_dir, f'diffusion_uncorrect_sup{iteri}.mp4'))
            save_video(diffusion_results_cur, os.path.join(self.opts.save_dir, f'diffusion_correct_sup{iteri}.mp4'))
            tensor_left = frames_vis.to(device)
            tensor_right = diffusion_results_cur.to(device)
            diffusion_results_cur = diffusion_results_cur.to(device)
            interval = torch.ones(T, self.opts.height, 30, 3, device=device)
            final_result = torch.cat((tensor_left, interval, tensor_right), dim=2)
            save_video(final_result, os.path.join(self.opts.save_dir, 'diffusion_sup1.mp4'), fps=self.opts.fps)

            
            # Segment foreground using sam2
            frames_list_cur = [(diffusion_results_cur[i] * 255).cpu().numpy().astype(np.uint8) for i in range(T)]
            template_mask = 1 - sam_masks_rotate[0, 0].cpu().numpy()
            if not hasattr(self, 'sampredictor'):
                self.setup_sam2()
            sam_frames = np.concatenate((diffusion_results.cpu().numpy()[:rotate_T], diffusion_results_cur.cpu().numpy()), axis=0)
            sam_labels = np.ones(self.sam_points.shape[0], dtype=np.int32)
            save_images_folder(sam_frames, os.path.join(self.opts.save_dir, "tmp_dir"))
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):            
                state = self.sampredictor.init_state(os.path.join(self.opts.save_dir, "tmp_dir"))
                for label_i in range(sam_labels.shape[0]):
                    sam_masks_cur, sam_painted_images = self.run_sam2_video((sam_frames * 255).astype(np.uint8), state, self.sam_points[label_i:label_i+1], sam_labels[label_i:label_i+1], None)
                    self.sampredictor.reset_state(state)
                    if label_i == 0:
                        sam_masks_sup1 = sam_masks_cur
                    else:
                        for t in range(len(sam_masks_cur)):
                            sam_masks_sup1[t] = (sam_masks_sup1[t] + sam_masks_cur[t]) > 0
            torch.cuda.empty_cache()
            gc.collect()

            sam_masks_sup1 = sam_masks_sup1[rotate_T:]
            sam_painted_images = sam_painted_images[rotate_T:]
            sam_masks_sup1 = 1 - torch.from_numpy(np.stack(sam_masks_sup1, axis=0)[...,None]).float().permute(0, 3, 1, 2) #[T, 1, H, W]
            sam_masks_sup1 = sam_masks_sup1.to(device)
            painted_images = torch.from_numpy(np.stack(sam_painted_images, axis=0))
            os.makedirs(os.path.join(self.opts.save_dir, f'sam_{iteri}'), exist_ok=True)
            save_video(
                (1-sam_masks_sup1.permute(0, 2, 3, 1).repeat(1, 1, 1, 3)),
                os.path.join(self.opts.save_dir, f'sam_{iteri}', 'masks.mp4'),
                fps=self.opts.fps,
            )
            save_video(
                painted_images / 255.0,
                os.path.join(self.opts.save_dir, f'sam_{iteri}', 'painted_images.mp4'),
                fps=self.opts.fps,
            )

            os.makedirs(os.path.join(self.opts.save_dir, f"images_{iteri}"), exist_ok=True)
            fn_list = [f"{i:05d}.png" for i in range(T)]
            #save_images_folder(frames_np, os.path.join(self.opts.save_dir, "images"))
            for i in range(T):
                img_cur = frames_list_cur[i]
                cv2.imwrite(os.path.join(self.opts.save_dir, f"images_{iteri}", fn_list[i]), cv2.cvtColor(img_cur, cv2.COLOR_RGB2BGR))

            # update occlusion relationship
            masks_pre = (1-sam_masks_sup1) * render_depth_masks_bg_full            
            masks_pre[masks_pre > 0] = 1.0
            depths_hw = render_depth_bg_full.view(T, H * W)
            for i in range(T):
                Ki_inv = K_inv[i]
                rays = (Ki_inv @ uv1[i].T).T               # (HW,3)
                Xi_cam = rays * depths_hw[i][:, None]      # (HW,3)

                R_c2w = Twcs_new[i][:3, :3]
                t_c2w = Twcs_new[i][:3, 3]
                Xi_world = (R_c2w @ Xi_cam.T).T + t_c2w[None]
                Xi_world = Xi_world.view(H, W, 3)

                points_world_pre[i] = Xi_world # (T,H,W,3)
            
            with self.aux_modules_offloaded():
                with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
                    _, occlusion_masks, _, _ = self.renderer.render(
                        c2ws=c2ws_target_cur_reverse,
                        Ks=intrinsic_target_cur,
                        points_world_from_img1=points_world_pre,
                        colors_from_img1=colors,
                        mask_img1=masks_pre,
                        normal_world_from_img1=None,
                        vis_threshold=getattr(self.opts, "vis_threshold", -0.1),
                    )
                    torch.cuda.empty_cache()
                    gc.collect()
            
            del (
                c2ws_target_cur,
                intrinsic_target_cur,
                c2ws_target_cur_reverse,

                render_depth_bg_full,
                render_depth_masks_bg_full,
                control_imgs_bg,
                render_masks_bg,
                control_imgs_fg,
                render_masks_fg,

                control_imgs, 
                render_masks, 
                view_masks,  

                diffusion_results1,
                diffusion_results_cur,
                final_result,
                sam_masks_cur,
                sam_painted_images,
                sam_masks_sup1,
                painted_images,

                depths_hw,
                Xi_cam,
                Xi_world,
            )
            torch.cuda.empty_cache()
            gc.collect()
    
    def align_video_depth(self, frame_list, prompt_result, align_mask):
        '''
        frame_list: list of frames
        prompt_result: metric depth from stream3r
        '''
        update_momentum = 0.99
        cache_scale_bias = None
        video_depth_model = VideoDepthAnythingDepthModel(model="vitl")
        video_depth_result: torch.Tensor = unpack_optional(
                video_depth_model.estimate(DepthEstimationInput(video_frame_list=frame_list)).relative_inv_depth
            )
        depth_masks = self.reliable_depth_mask_range_batch(video_depth_result.unsqueeze(1).reciprocal(), window_size=5, ratio_thresh=0.1).squeeze()  # (T,1,H,W)

        video_inv_depth_total = []
        for frame_idx, frame in enumerate(frame_list):
            video_depth_inv_depth = video_depth_result[frame_idx]
            mask_cur = align_mask[frame_idx].cpu().numpy()
            mask_cur = cv2.erode(mask_cur, np.ones((5, 5)), iterations=2).astype(np.uint8)
            mask_cur = torch.from_numpy(mask_cur).to(align_mask.device)

            sparse_mask = mask_cur * depth_masks[frame_idx] * (video_depth_inv_depth > 1e-3)
            keep_ratio = 0.25                                 # keep 25% mask
            idx = torch.where(sparse_mask)
            n_keep = int(idx[0].numel() * keep_ratio)
            if n_keep > 0:
                perm = torch.randperm(idx[0].numel(), device=sparse_mask.device)[:n_keep]
                sparse_mask = torch.zeros_like(sparse_mask)
                sparse_mask[idx[0][perm], idx[1][perm]] = True

            try:
                _, scale, bias = align_inv_depth_to_depth(
                    unpack_optional(video_depth_inv_depth),
                    prompt_result[frame_idx],
                    sparse_mask,
                    False,
                )
            except RuntimeError:
                scale, bias = cache_scale_bias

            
            if cache_scale_bias is None:
                cache_scale_bias = (scale, bias)
            scale = cache_scale_bias[0] * update_momentum + scale * (1 - update_momentum)
            bias = cache_scale_bias[1] * update_momentum + bias * (1 - update_momentum)
            cache_scale_bias = (scale, bias)

            video_inv_depth = video_depth_inv_depth * scale + bias
            video_inv_depth[video_inv_depth < 1e-3] = 1e-3
            video_inv_depth = video_inv_depth.reciprocal()

            video_inv_depth_total.append(video_inv_depth)
        
        self._safe_move_module(getattr(video_depth_model, "model", None), "cpu")
        del video_depth_model

        torch.cuda.empty_cache()
        gc.collect()

        video_inv_depth = torch.stack(video_inv_depth_total)
        return video_inv_depth

