import sys
import os
from pathlib import Path
from typing import Tuple, List
from huggingface_hub import hf_hub_download

# Renamed from demo_dynamic_stream3r_vda_mask.py

# Make repo-local modules importable even when launched outside repo root.
# NOTE: Keep repo root at sys.path[0] to avoid name collisions (e.g. MoGe ships a top-level `utils` package).
_REPO_ROOT = Path(__file__).parent.resolve()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Make extern deps importable (after repo root).
# - `moge` is vendored at `extern/moge` (a proper python package), so we add `extern/` to import it as `import moge`.
# - `sam2` and `vipe` require adding their repo roots (they contain the actual python packages inside).
for _rel in ("extern/STream3R", "extern/vipe", "extern/sam2", "extern"):
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

# align depth with VDA
from vipe.priors.depth.videodepthanything import VideoDepthAnythingDepthModel
from vipe.priors.depth import DepthEstimationInput
from vipe.utils.misc import unpack_optional
from vipe.priors.depth.alignment import align_inv_depth_to_depth

# single view packages (kept the same as demo_dynamic_vipe for downstream rendering flow)
from carvekit.ml.wrap.tracer_b7 import TracerUniversalB7
from diffusers.utils import export_to_video

from utils.utils import traj_map, points_padding, np_points_padding, set_initial_camera, build_cameras
from datetime import datetime
from moge.model.v2 import MoGeModel  # Let's try MoGe-2

from utils.warp_utils import *  # read_video_frames, save_video, etc.
import hydra
from utils.imagesplatrender import BiSplatRenderer

# other packages
import shutil
import einops
import warnings
import torch
import math
from contextlib import contextmanager
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
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R
import re

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
from utils.tensor_stream import _preprocess_frames_for_stream3r


class UniScene:
    def __init__(self, opts, gradio: bool = False):
        self.opts = opts
        self._apply_weight_source_policy()
        if not hasattr(self.opts, "base_height"):
            self.opts.base_height = int(getattr(self.opts, "height", 480))
        if not hasattr(self.opts, "base_width"):
            self.opts.base_width = int(getattr(self.opts, "width", 832))
        if not hasattr(self.opts, "advanced_render"):
            self.opts.advanced_render = True
        self.device = opts.device
        self.setup_stream3r()
        self.setup_moge()
        self.seg_net = TracerUniversalB7(device=self.opts.device, batch_size=1, model_path=self.opts.segnet_path).eval()
        self.caption_processor = AutoProcessor.from_pretrained(opts.blip_path)
        self.captioner = Blip2ForConditionalGeneration.from_pretrained(opts.blip_path, torch_dtype=torch.float16).to(opts.device)
        self.renderer = BiSplatRenderer(resolution=(opts.height, opts.width), device=opts.device)
        self.sampredictor = None
        self.setup_diffusion()
        self.offload_aux_modules_to_cpu()

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

        if _is_default_local_path(getattr(self.opts, "transformer_path", "")):
            self.opts.transformer_path = hf_transformer_repo
        if _is_default_local_path(getattr(self.opts, "model_name", "")):
            self.opts.model_name = hf_wan_repo
        if _is_default_local_path(getattr(self.opts, "blip_path", "")):
            self.opts.blip_path = hf_blip_repo
        if _is_default_local_path(getattr(self.opts, "stream3r_path", "")):
            self.opts.stream3r_path = hf_stream3r_repo
        if _is_default_local_path(getattr(self.opts, "moge_path", "")):
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

    def _refresh_gradio_save_dir(self) -> None:
        prefix = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.opts.save_dir = os.path.join("./output/gradio", prefix)
        os.makedirs(self.opts.save_dir, exist_ok=True)

    def setup_stream3r(self):
        # Load pretrained STream3R once
        self.stream3r = STream3R.from_pretrained(self.opts.stream3r_path).to(self.opts.device).eval()

    def setup_moge(self):
        self.depth_model = MoGeModel.from_pretrained(self.opts.moge_path).to(self.opts.device).eval()
    
    def setup_sam2(self):
        if self.sampredictor is None:
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

    def nvs_single_view(self, image=None):
        with self.active_aux_modules():
            if image is None:
                image = Image.open(self.opts.image_dir).convert("RGB")
            elif not isinstance(image, Image.Image):
                image = Image.fromarray(image).convert("RGB")

            image = self._apply_input_resize_policy(image)

            validation_image = ToTensor()(image)[None].to(self.opts.device)  # [1,c,h,w], 0~1
            depth_image = validation_image[0]
            depth, masks, normal, intrinsic, K, K_inv = self.run_moge(depth_image)

            # get pointcloud
            points2d = torch.stack(
                torch.meshgrid(
                    torch.arange(self.opts.width, dtype=torch.float32),
                    torch.arange(self.opts.height, dtype=torch.float32),
                    indexing="xy",
                ),
                -1,
            ).to(self.opts.device)  # [h,w,2]
            points3d = points_padding(points2d).reshape(self.opts.height * self.opts.width, 3)  # [hw,3]
            points3d = (K_inv @ points3d.T * depth.reshape(1, self.opts.height * self.opts.width)).T
            colors = (depth_image * 255).to(torch.uint8).permute(1, 2, 0).reshape(self.opts.height * self.opts.width, 3)

            points3d_np = points3d.detach().cpu().numpy()
            colors_np = colors.detach().cpu().numpy()
            normal_np = normal.detach().cpu().numpy()

            # inference foreground mask
            with torch.no_grad():
                origin_w_, origin_h_ = image.size
                image_pil = image.resize((512, 512))
                fg_mask = self.seg_net([image_pil])[0]
                fg_mask = fg_mask.resize((origin_w_, origin_h_))

            fg_mask_np = np.array(fg_mask, dtype=np.float32)
            fg_mask_bool = fg_mask_np > 0.5
            if fg_mask_bool.mean() < 0.05:
                fg_mask_bool[...] = True
            fg_mask_tensor = torch.from_numpy(fg_mask_bool.astype(np.float32)).to(self.opts.device)
            depth_avg = torch.median(depth[0, 0][torch.from_numpy(fg_mask_bool).to(self.opts.device, torch.bool)]).item()
            w2c_0, c2w_0 = set_initial_camera(self.opts.elevation, depth_avg)

        c2w_0_np = c2w_0.detach().cpu().numpy()
        points3d_np = (c2w_0_np[:3] @ np_points_padding(points3d_np).T).T
        normal_np = (c2w_0_np[:3, :3] @ normal_np.T).T
        points_world_tensor = torch.from_numpy(points3d_np.reshape(self.opts.height, self.opts.width, 3)).to(self.opts.device, torch.float32)
        colors_tensor = torch.from_numpy(((colors_np / 255.0) * 2.0 - 1.0).reshape(self.opts.height, self.opts.width, 3)).to(self.opts.device, torch.float32)
        normals_tensor = torch.from_numpy(normal_np.reshape(self.opts.height, self.opts.width, 3).astype(np.float32)).to(self.opts.device, torch.float32)
        mask_reliable_tensor = masks.view(self.opts.height, self.opts.width).to(self.opts.device, torch.float32)

        # 生成轨迹/相机
        if self.opts.traj_type == "custom":
            cam_traj, x_offset, y_offset, z_offset, d_theta, d_phi, d_r = (
                "free",
                self.opts.x_offset,
                self.opts.y_offset,
                self.opts.z_offset,
                self.opts.d_theta,
                self.opts.d_phi,
                self.opts.d_r,
            )
        else: 
            cam_traj, x_offset, y_offset, z_offset, d_theta, d_phi, d_r = traj_map(self.opts.traj_type)

        w2cs, c2ws, intrinsic = build_cameras(
                cam_traj=cam_traj,
                w2c_0=w2c_0,
                c2w_0=c2w_0,
                intrinsic=intrinsic,
                nframe=self.opts.video_length,
                focal_length=self.opts.focal_length,
                d_theta=d_theta,
                d_phi=d_phi,
                d_r=d_r,
                radius=depth_avg,
                x_offset=x_offset,
                y_offset=y_offset,
                z_offset=z_offset,
            )
        w2cs_reverse, c2ws_reverse, _ = build_cameras(
            cam_traj=cam_traj,
            w2c_0=w2c_0,
            c2w_0=c2w_0,
            intrinsic=intrinsic,
            nframe=self.opts.video_length,
            focal_length=self.opts.focal_length,
            d_theta=-d_theta,
            d_phi=-d_phi,
            d_r=d_r,
            radius=depth_avg,
            x_offset=-x_offset,
            y_offset=-y_offset,
            z_offset=-z_offset,
        )

        device = self.opts.device
        w2c_0 = w2c_0.to(device)
        c2w_0 = c2w_0.to(device)
        w2cs = w2cs.to(device)
        c2ws = c2ws.to(device)
        intrinsic = intrinsic.to(device)
        w2cs_reverse = w2cs_reverse.to(device)
        c2ws_reverse = c2ws_reverse.to(device)
        K = K.to(device)
        depth = depth.to(device)

        render_method = str(getattr(self.opts, "render_method", "hybrid")).strip().lower()
        if render_method not in {"warp", "hybrid", "mesh"}:
            raise ValueError(f"Invalid render_method: {render_method!r} (expected 'warp', 'hybrid' or 'mesh')")

        if abs(d_phi) > 180:
            self.opts.warp_with_occlusion = False

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            if render_method == "hybrid":
                try:
                    from utils.meshrender import MeshWarper
                    from utils.pointcloud import run_render
                except ModuleNotFoundError as exc:
                    if getattr(exc, "name", None) == "pytorch3d":
                        raise ModuleNotFoundError(
                            "render_method='hybrid' requires PyTorch3D, but it is not installed.\n"
                            "Install PyTorch3D (from source) following README.md, or switch to "
                            "`--render_method warp`."
                        ) from exc
                    raise

                T = self.opts.video_length
                meshwarp = MeshWarper(resolution=(self.opts.height, self.opts.width), device=str(device))

                pcd = points_world_tensor.unsqueeze(0)
                imgs = colors_tensor.unsqueeze(0)
                masks_rgb = mask_reliable_tensor.view(1, self.opts.height, self.opts.width, 1).repeat(1, 1, 1, 3)

                warped_images2 = []
                warped_images1 = []
                masks1 = []

                for i in range(T):
                    _, _, warped_depth_pcd = run_render(
                        pcd=pcd,
                        imgs=imgs,
                        masks=masks_rgb,
                        H=self.opts.height,
                        W=self.opts.width,
                        c2ws=c2ws[i : i + 1],
                        K=intrinsic[i : i + 1],
                        num_views=1,
                        return_mask=True,
                        return_depth=True,
                        device=device,
                    )
                    warped_frame2, _, warped_depth_mesh = meshwarp.forward_warp(
                        imgs.permute(0, 3, 1, 2),
                        depth,
                        w2c_0.unsqueeze(0),
                        w2cs[i : i + 1],
                        K.unsqueeze(0),
                        intrinsic[i : i + 1],
                    )
                    warped_images2.append(warped_frame2)

                    warped_mask_rgb, _, _ = meshwarp.forward_warp(
                        mask_reliable_tensor.view(1, 1, self.opts.height, self.opts.width).repeat(1, 3, 1, 1) * 2 - 1,
                        depth,
                        w2c_0.unsqueeze(0),
                        w2cs[i : i + 1],
                        K.unsqueeze(0),
                        intrinsic[i : i + 1],
                    )
                    warped_images1.append(warped_mask_rgb)
                    masks1.append((warped_depth_pcd.to(warped_depth_mesh.device) <= warped_depth_mesh).to(torch.float32))

                cond_video = (torch.cat(warped_images2) + 1.0) / 2.0
                cond_video1 = (torch.cat(warped_images1) + 1.0) / 2.0
                cond_video1 = (cond_video1 >= 0.5).float()
                cond_masks1 = torch.cat(masks1)

                control_imgs = cond_video * cond_video1 * cond_masks1
                render_masks = cond_video1[:, 0:1] * cond_masks1
                control_imgs = control_imgs * 2.0 - 1.0

            elif render_method == "mesh":
                try:
                    from utils.meshrenderex import MeshWarperEx
                except ModuleNotFoundError as exc:
                    if getattr(exc, "name", None) == "pytorch3d":
                        raise ModuleNotFoundError(
                            "render_method='mesh' requires PyTorch3D, but it is not installed.\n"
                            "Install PyTorch3D (from source) following README.md, or switch to "
                            "`--render_method warp`."
                        ) from exc
                    raise

                T = self.opts.video_length
                meshwarp = MeshWarperEx(resolution=(self.opts.height, self.opts.width), device=str(device))

                imgs = colors_tensor.unsqueeze(0)
                whitemesh = torch.ones_like(imgs)

                warped_images2 = []
                warped_images1 = []

                for i in range(T):
                    warped_frame2, _, _ = meshwarp.forward_warp(
                        imgs.permute(0, 3, 1, 2),
                        depth,
                        w2c_0.unsqueeze(0),
                        w2cs[i : i + 1],
                        K.unsqueeze(0),
                        intrinsic[i : i + 1],
                    )
                    warped_images2.append(warped_frame2)

                    warped_frame1, _, _ = meshwarp.forward_warp(
                        whitemesh.permute(0, 3, 1, 2),
                        depth,
                        w2c_0.unsqueeze(0),
                        w2cs[i : i + 1],
                        K.unsqueeze(0),
                        intrinsic[i : i + 1],
                    )
                    warped_images1.append((warped_frame1 > 0.0).float())

                cond_video = (torch.cat(warped_images2) + 1.0) / 2.0
                cond_masks = torch.cat(warped_images1)[:, 0:1]
                control_imgs = cond_video * 2.0 - 1.0
                render_masks = cond_masks

            elif self.opts.warp_with_occlusion:
                T = self.opts.video_length
                points_world_rep = points_world_tensor.unsqueeze(0).repeat(T, 1, 1, 1)
                colors_rep = colors_tensor.unsqueeze(0).repeat(T, 1, 1, 1)
                normals_rep = normals_tensor.unsqueeze(0).repeat(T, 1, 1, 1)
                mask_reliable_rep = mask_reliable_tensor.view(1, 1, self.opts.height, self.opts.width).repeat(T, 1, 1, 1)
                fg_mask_rep = fg_mask_tensor.view(1, 1, self.opts.height, self.opts.width).repeat(T, 1, 1, 1)
                points_world_input = points_world_rep.clone()
                colors_input = colors_rep.clone()
                masks_input = mask_reliable_rep.clone()

                control_imgs_fg, render_masks_fg, render_depth_bg, render_depth_masks_bg = self.renderer.render(
                    c2ws=c2ws,
                    Ks=intrinsic,
                    points_world_from_img1=points_world_rep,
                    colors_from_img1=colors_rep,
                    mask_img1=mask_reliable_rep, # * (1.0 - fg_mask_rep),
                    normal_world_from_img1=None,
                    vis_threshold=getattr(self.opts, "vis_threshold", -0.1),
                )

                save_video(render_masks_fg.permute(0,2,3,1).repeat(1, 1, 1, 3), os.path.join(self.opts.save_dir, 'mask_inconsis.mp4'), fps=self.opts.fps)
                #depths_hw = render_depth_bg.view(T, self.opts.height * self.opts.width)
                for i in range(T):
                    points3d = points_padding(points2d).reshape(self.opts.height * self.opts.width, 3)  # [hw,3]
                    points3d = (K_inv @ points3d.T * render_depth_bg[i].reshape(1, self.opts.height * self.opts.width)).T
                    points3d_np = points3d.detach().cpu().numpy()
                    points3d_np = (c2w_0_np[:3] @ np_points_padding(points3d_np).T).T
                    points_world_tensor = torch.from_numpy(points3d_np.reshape(self.opts.height, self.opts.width, 3)).to(self.opts.device, torch.float32)

                    points_world_input[i] = points_world_tensor
                    colors_input[i] = control_imgs_fg[i].permute(1, 2, 0)
                    masks_input[i] = render_masks_fg[i]
                
                _, _, _, render_masks_occ = self.renderer.render(
                    c2ws=c2ws_reverse,
                    Ks=intrinsic,
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
                    points_world_input[i] = points_world_rep[i]
                    colors_input[i] = colors_rep[i]

                masks_input[masks_input>0] = 1
                masks_input *= mask_reliable_rep
                control_imgs, render_masks, _, _ = self.renderer.render(
                    c2ws=c2ws,
                    Ks=intrinsic,
                    points_world_from_img1=points_world_input,
                    colors_from_img1=colors_input,
                    mask_img1=masks_input,
                    normal_world_from_img1=normals_rep,
                    vis_threshold=getattr(self.opts, "vis_threshold", -0.1),
                )
            else:
                control_imgs, render_masks, _, _ = self.renderer.render(
                    c2ws=c2ws,
                    Ks=intrinsic,
                    points_world_from_img1=points_world_tensor.unsqueeze(0).repeat(self.opts.video_length, 1, 1, 1),
                    colors_from_img1=colors_tensor.unsqueeze(0).repeat(self.opts.video_length, 1, 1, 1),
                    mask_img1=mask_reliable_tensor.view(1, 1, self.opts.height, self.opts.width).repeat(self.opts.video_length, 1, 1, 1),
                    normal_world_from_img1=normals_tensor.unsqueeze(0).repeat(self.opts.video_length, 1, 1, 1),
                )

            control_imgs[0:1] = validation_image * 2.0 - 1.0
            render_masks[0:1] = 1.0

            render_results = einops.rearrange(control_imgs, "f c h w -> f h w c", f=self.opts.video_length)
            view_masks = einops.rearrange(render_masks, "f c h w -> f h w c", f=self.opts.video_length)
            view_masks = 1.0 - view_masks

            render_results = (render_results + 1.0) / 2.0
            save_video(render_results, os.path.join(self.opts.save_dir, "render.mp4"))
            save_video(view_masks.repeat(1, 1, 1, 3), os.path.join(self.opts.save_dir, "mask.mp4"))

            render_results = render_results * 2.0 - 1.0

            prompt = self.get_caption(image)
            print(prompt)

            diffusion_results1 = self.run_diffusion(render_results, view_masks, prompt, ref_video=None)
            diffusion_results2 = self.color_correct(diffusion_results1, (render_results + 1.0) / 2.0)
            save_video(diffusion_results1, os.path.join(self.opts.save_dir, "diffusion_uncorrect.mp4"))
            save_video(diffusion_results2, os.path.join(self.opts.save_dir, "diffusion_correct.mp4"))
            return diffusion_results1

    def run_single_view_gradio(self, i2v_input_image, i2v_elevation, i2v_center_scale, i2v_pose, i2v_steps, i2v_seed, i2v_guidance_scale, vis_threshold):
        self._refresh_gradio_save_dir()

        self.opts.ddim_steps = i2v_steps
        try:
            self.opts.diffusion_guidance_scale = float(i2v_guidance_scale)
        except Exception:
            self.opts.diffusion_guidance_scale = getattr(self.opts, "diffusion_guidance_scale", 1.0)

        self.opts.elevation = float(i2v_elevation)
        self.opts.radius_scale = i2v_center_scale
        pose_text = str(i2v_pose).strip()
        if pose_text.lower().startswith("swing"):
            self.opts.traj_type = "swing1"
            self.opts.d_phi = 0.0
            self.opts.d_theta = 0.0
            self.opts.x_offset = 0.0
            self.opts.y_offset = 0.0
            self.opts.z_offset = 0.0
        else:
            self.opts.traj_type = "custom"
            vals = [float(x) for x in pose_text.replace(",", ";").split(";") if x.strip() != ""]
            while len(vals) < 5:
                vals.append(0.0)
            self.opts.d_phi, self.opts.d_theta, self.opts.x_offset, self.opts.y_offset, self.opts.z_offset = vals[:5]

        image = Image.fromarray(i2v_input_image).convert("RGB")

        self.opts.vis_threshold = float(vis_threshold)
        self.nvs_single_view(image)
        gen_dir = os.path.join(self.opts.save_dir, "diffusion_uncorrect.mp4")
        return gen_dir

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
                # Use .to() instead of .cuda() to respect CUDA_VISIBLE_DEVICES
                module.to(device_str)
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
        self.captioner = self.captioner.to(self.opts.device, torch.float16)
        generated_ids = self.captioner.generate(**inputs)
        generated_text = self.caption_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return generated_text + self.opts.refine_prompt

    def nvs_dynamic_view(self):
        with self.active_aux_modules():
            self._nvs_dynamic_view_impl()

    def _nvs_dynamic_view_impl(self):
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

        # 1) Read video frames (numpy float32 in [0,1])
        frames_np = read_video_frames(
            self.opts.image_dir,
            self.opts.video_length,
            self.opts.stride,
            self.opts.max_res,
            width=int(self.opts.width),
            height=int(self.opts.height),
            center_crop=not bool(getattr(self.opts, "keep_aspect_ratio", False)),
        )
        T, H, W, _ = frames_np.shape
        assert T == self.opts.video_length, "读取到的帧数与期望不一致"

        render_method = str(getattr(self.opts, "render_method", "hybrid")).strip().lower()
        if render_method not in {"warp", "hybrid", "mesh"}:
            raise ValueError(f"Invalid render_method: {render_method!r} (expected 'warp', 'hybrid' or 'mesh')")

        align_with_vda = bool(getattr(self.opts, "align_with_vda", False))
        warp_with_occlusion = bool(getattr(self.opts, "warp_with_occlusion", False))
        if render_method in {"hybrid", "mesh"}:
            warp_with_occlusion = False

        traj_type_for_check = str(getattr(self.opts, "traj_type", "custom")).strip().lower()
        d_phi_for_check = float(getattr(self.opts, "d_phi", 0.0))
        if traj_type_for_check != "custom":
            try:
                _, _, _, _, _, d_phi_for_check, _ = traj_map(traj_type_for_check)
            except Exception:
                pass
        if abs(d_phi_for_check) > 180:
            warp_with_occlusion = False

        if not bool(getattr(self.opts, "advanced_render", True)):
            align_with_vda = False
            warp_with_occlusion = False

        self.opts.align_with_vda = align_with_vda
        self.opts.warp_with_occlusion = warp_with_occlusion
        self.opts.advanced_render = align_with_vda or warp_with_occlusion

        # 2) STream3R inference: per-frame pose enc, depth, intrinsics
        imgs_stream = _preprocess_frames_for_stream3r(frames_np, mode="crop").to(device)  # (T,3,Hs,Ws)
        Hs, Ws = imgs_stream.shape[-2:]

        with torch.no_grad():
            # 默认采用最高精度的全局注意力模式（full）
            outputs = self.stream3r(imgs_stream, mode="full")

        # pose encoding -> extrinsic/intrinsic at STream3R input resolution
        pose_enc = outputs["pose_enc"]  # (1,T,9)

        # Smoothing utilities to reduce jitter
        def smooth_quaternions(q_in: torch.Tensor, sigma: float = 0.8) -> torch.Tensor:
            """Gaussian smooth rotations via rotvec (axis-angle) + SLERP behavior.

            Procedure:
            - Enforce quaternion sign continuity (xyzw, scalar-last).
            - Convert to rotvec with scipy Rotation.
            - Apply gaussian_filter1d along time on rotvec.
            - Convert back to unit quaternions.

            Args:
                q_in: Tensor (T,4) or (1,T,4)
                sigma: Gaussian sigma for smoothing in frame units

            Returns:
                Tensor (T,4) smoothed unit quaternions on the same device/dtype
            """
            if q_in.dim() == 3 and q_in.shape[0] == 1:
                q = q_in.squeeze(0).detach().cpu().numpy().copy()
            elif q_in.dim() == 2:
                q = q_in.detach().cpu().numpy().copy()
            else:
                raise ValueError(f"Unexpected quaternion shape: {tuple(q_in.shape)}")

            Tq = q.shape[0]
            if Tq <= 1:
                return q_in.squeeze(0) if q_in.dim() == 3 else q_in

            # Ensure sign continuity
            for t in range(1, Tq):
                if (q[t - 1] @ q[t]) < 0.0:
                    q[t] = -q[t]

            # Convert to rotvec (axis-angle), smooth, then convert back
            rot = R.from_quat(q)           # xyzw
            rv = rot.as_rotvec()           # (T,3)
            rv_s = gaussian_filter1d(rv, sigma=sigma, axis=0, mode='nearest')
            rot_s = R.from_rotvec(rv_s)
            q_s = rot_s.as_quat()          # xyzw
            q_s = q_s / (np.linalg.norm(q_s, axis=1, keepdims=True) + 1e-12)

            q_s_t = torch.from_numpy(q_s).to(q_in.device).to(q_in.dtype)
            return q_s_t

        def smooth_translation(t_in: torch.Tensor, sigma: float = 0.8) -> torch.Tensor:
            """Gaussian smooth a sequence of translations along time.

            Args:
                t_in: Tensor (T,3) or (1,T,3)
                sigma: Gaussian sigma in frame units
            Returns:
                Tensor (T,3)
            """
            if t_in.dim() == 3 and t_in.shape[0] == 1:
                t = t_in.squeeze(0).detach().cpu().numpy()
            elif t_in.dim() == 2:
                t = t_in.detach().cpu().numpy()
            else:
                raise ValueError(f"Unexpected translation shape: {tuple(t_in.shape)}")

            if t.shape[0] <= 1:
                return t_in.squeeze(0) if t_in.dim() == 3 else t_in
            
            t_smooth = gaussian_filter1d(t, sigma=sigma, axis=0, mode='nearest')
            t_smooth_t = torch.from_numpy(t_smooth).to(t_in.device).to(t_in.dtype)
            return t_smooth_t

        def smooth_pose_encoding_full(pose_enc_in: torch.Tensor, sigma_all: float = 3.0) -> torch.Tensor:
            """Smooth full pose encoding (Txyz, quat, FoV) with one sigma.

            - Translation: gaussian along time
            - Rotation: rotvec gaussian along time with sign continuity
            - FoV: gaussian along time with clamp to (0, pi)

            Args:
                pose_enc_in: Tensor (1,T,9)
                sigma_all: non-negative sigma for all components
            Returns:
                Smoothed pose encoding (1,T,9)
            """
            if sigma_all <= 0.0:
                return pose_enc_in
            assert pose_enc_in.dim() == 3 and pose_enc_in.shape[0] == 1 and pose_enc_in.shape[-1] == 9
            Tlen = pose_enc_in.shape[1]
            if Tlen <= 1:
                return pose_enc_in

            pe = pose_enc_in.clone()
            # Smooth translation
            trans = pe[0, :, :3]
            trans_s = smooth_translation(trans, sigma=sigma_all)
            pe[0, :, :3] = trans_s

            # Smooth rotation
            quat = pe[0, :, 3:7]
            quat_s = smooth_quaternions(quat, sigma=sigma_all)
            pe[0, :, 3:7] = quat_s

            # Smooth FoV
            fov = pe[0, :, 7:9].detach().cpu().numpy()
            fov_s = gaussian_filter1d(fov, sigma=sigma_all, axis=0, mode='nearest')
            fov_s = np.clip(fov_s, 1e-3, np.pi - 1e-3)
            pe[0, :, 7:9] = torch.from_numpy(fov_s).to(pe.device).to(pe.dtype)

            return pe

        do_smooth = False
        if do_smooth:
            # Unified smoothing: translation, rotation, FoV all with sigma=3
            pose_enc = smooth_pose_encoding_full(pose_enc, sigma_all=3.0)

        # Fix FoV to first frame across the whole sequence (no zoom jitter)
        fov0 = pose_enc[0, 0, 7:9].clone()
        pose_enc[0, :, 7:9] = fov0.unsqueeze(0).expand(pose_enc.shape[1], -1)

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

        # Note: Depth temporal smoothing removed per request; only camera is smoothed.

        # scale intrinsics to original resolution
        scale_x = float(W) / float(Ws)
        scale_y = float(H) / float(Hs)
        K_3x3 = intri_33.clone()
        K_3x3[:, 0, 0] *= scale_x
        K_3x3[:, 1, 1] *= scale_y
        K_3x3[:, 0, 2] *= scale_x
        K_3x3[:, 1, 2] *= scale_y
        K_inv = torch.linalg.inv(K_3x3)

        del imgs_stream, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Depth ready for downstream (no VDA alignment)
            # Convert to c2w (cam->world) and align to our world using first frame and initial camera
        Twcs_stream = closed_form_inverse_se3(extri_34)  # (T,4,4)

        '''# Align fg depth use VDA
        frames_list_float = [
            (frames_np[i]) for i in range(T)
        ]
        depths_t = self.align_video_depth(frames_list_float, depths_t, 1-sam_masks.squeeze())'''

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
        del frames_tensor, normals_cam_list
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 3.1) segment foreground
                # 分割前景物体
        # 找到fg_sum_list 中最大的值，作为关键帧
        if self.opts.advanced_render:
            self.setup_sam2()
            key_frame_idx = 0
            print('KeyFrame: ',key_frame_idx)

            # motion_fg_mask = fg_mask0
            motion_fg_mask = fg_mask0
            # numpy opencv erode
            motion_fg_mask_ero = motion_fg_mask.cpu().numpy().astype(np.uint8)
            motion_fg_mask_ero = cv2.erode(motion_fg_mask_ero, np.ones((9, 9), np.uint8), iterations=2)
            motion_fg_mask_ero = torch.from_numpy(motion_fg_mask_ero).to(device).type(torch.float32)

            sample_interval = 100
            sample_mask = np.zeros((1, H, W), dtype=np.uint8)
            sample_mask[:, ::sample_interval, ::sample_interval] = 1
            sample_mask = torch.from_numpy(sample_mask).to(device)
            print(motion_fg_mask.shape)
            print(sample_mask.shape)
            sample_motion_fg_mask = motion_fg_mask_ero * sample_mask
            # 如果有效点太少，补充几个点
            while sample_interval > 10:
                if sample_motion_fg_mask.sum() < 10:
                    sample_interval = sample_interval - 10
                    sample_mask = np.zeros((1, H, W), dtype=np.uint8)
                    sample_mask[:, ::sample_interval, ::sample_interval] = 1
                    sample_mask = torch.from_numpy(sample_mask).to(device)
                    sample_motion_fg_mask = motion_fg_mask_ero * sample_mask
                else:
                    break
            mask_idx = torch.nonzero(sample_motion_fg_mask[0], as_tuple=False)  # → (N, 2) LongTensor
            sam_points = mask_idx[:, [1, 0]].int().cpu().numpy()
            sam_labels = np.ones(sam_points.shape[0], dtype=np.int32)
            
            self.save_images_folder(frames_np[key_frame_idx*10:], os.path.join(self.opts.save_dir, "tmp_dir"))
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):            
                for label_i in range(sam_labels.shape[0]):
                    state = self.sampredictor.init_state(os.path.join(self.opts.save_dir, "tmp_dir"))
                    sam_masks_cur, sam_painted_images = self.run_sam2_video((frames_np * 255).astype(np.uint8), state, sam_points[label_i:label_i+1], sam_labels[label_i:label_i+1], None)
                    if label_i == 0:
                        sam_masks = sam_masks_cur
                    else:
                        for t in range(len(sam_masks_cur)):
                            sam_masks[t] = (sam_masks[t] + sam_masks_cur[t]) > 0
                    del state, sam_masks_cur
            
            sam_masks = 1 - torch.from_numpy(np.stack(sam_masks, axis=0)[...,None]).float().permute(0, 3, 1, 2) #[T, 1, H, W]
            sam_masks = sam_masks.to(device)
            painted_images = torch.from_numpy(np.stack(sam_painted_images, axis=0))
            os.makedirs(os.path.join(self.opts.save_dir, 'sam'), exist_ok=True)
            print(painted_images.shape)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
            torch.cuda.empty_cache()

            if self.opts.align_with_vda:
                # 3.2) align fg depth with VDA
                frames_list_float = [
                    (frames_np[i]) for i in range(T)
                ]
                aligned_depths_t = self.align_video_depth(frames_list_float, depths_t, 1-sam_masks.squeeze())
                print("depth_bg.shape:", aligned_depths_t.shape)
                #depth_fg = (1-sam_masks.squeeze()) * aligned_depths_t
                #viz_depth_list([depth_fg[i].cpu().numpy() for i in range(T)], self.opts.save_dir + "/depth_fg.mp4")
                
                ### dilate
                sam_masks_dilate = 1-sam_masks.squeeze().cpu().numpy()
                print(sam_masks_dilate.shape)
                for i in range(sam_masks_dilate.shape[0]):
                    sam_masks_dilate[i] = cv2.dilate(sam_masks_dilate[i], np.ones((9, 9), np.uint8), iterations=3)
                sam_masks_dilate = torch.from_numpy(sam_masks_dilate).to(device).type(torch.float32)
                depths_t = (1-sam_masks_dilate) * depths_t + (1-sam_masks.squeeze()) * aligned_depths_t
                #max_depth = torch.max(depths_t)
                #depths_t_viz = depth_masks * depths_t + ~depth_masks*max_depth
                #depths_t = aligned_depths_t.to(device)
                #viz_depth_list([depths_t_viz[i].cpu().numpy() for i in range(T)], self.opts.save_dir + "/depth_align.mp4")


        # 4) Unproject each frame to world points via our own math (consistent with existing pipeline)
        uu, vv = torch.meshgrid(
            torch.arange(W, device=device, dtype=torch.float32),
            torch.arange(H, device=device, dtype=torch.float32),
            indexing="xy",
        )
        uv1 = torch.stack([uu, vv, torch.ones_like(uu)], dim=-1).view(1, H * W, 3).repeat(T, 1, 1)  # (T,HW,3)
        depths_hw = depths_t.view(T, H * W)

        points_world_list = []
        colors_list = []
        mask_list = []

        ratio_thresh = float(getattr(self.opts, "depth_reliable_ratio_thresh", 0.1))
        depth_masks = self.reliable_depth_mask_range_batch(
            depths_t.unsqueeze(1), window_size=5, ratio_thresh=ratio_thresh
        )  # (T,1,H,W)
        frames_torch = torch.from_numpy(frames_np).to(device).to(torch.float32)  # (T,H,W,3) in [0,1]

        for i in range(T):
            Ki_inv = K_inv[i]
            rays = (Ki_inv @ uv1[i].T).T               # (HW,3)
            Xi_cam = rays * depths_hw[i][:, None]      # (HW,3)

            R_c2w = Twcs_new[i][:3, :3]
            t_c2w = Twcs_new[i][:3, 3]
            Xi_world = (R_c2w @ Xi_cam.T).T + t_c2w[None]
            Xi_world = Xi_world.view(H, W, 3)

            points_world_list.append(Xi_world)
            colors_list.append(frames_torch[i] * 2.0 - 1.0)
            mask_list.append(depth_masks[i])

        points_world = torch.stack(points_world_list, dim=0)  # (T,H,W,3)
        colors = torch.stack(colors_list, dim=0)              # (T,H,W,3)
        masks = torch.stack(mask_list, dim=0).float()         # (T,1,H,W)

        # 5) Build target camera trajectory and intrinsics (reuse existing util)
        if self.opts.traj_type == "custom":
            cam_traj, x_offset, y_offset, z_offset, d_theta, d_phi, d_r = (
                "free",
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
        c2w_0_cpu = c2w_0.detach().cpu()
        w2c_0_cpu = torch.linalg.inv(c2w_0_cpu)
        w2cs_target, c2ws_target, intrinsic_target = build_cameras(
            cam_traj=cam_traj,
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
        w2cs_target = w2cs_target.to(device)

        # 6) Render
        if abs(d_phi) > 180:
            self.opts.warp_with_occlusion = False

        if render_method == "hybrid":
            try:
                from utils.meshrender import MeshWarper
                from utils.pointcloud import run_render
            except ModuleNotFoundError as exc:
                if getattr(exc, "name", None) == "pytorch3d":
                    raise ModuleNotFoundError(
                        "render_method='hybrid' requires PyTorch3D, but it is not installed.\n"
                        "Install PyTorch3D (from source) following README.md, or switch to "
                        "`--render_method warp`."
                    ) from exc
                raise

            warped_images2 = []
            warped_images1 = []
            masks1 = []

            meshwarp = MeshWarper(resolution=(H, W), device=str(device))
            depths_mesh = depths_t
            if self.opts.advanced_render and self.opts.align_with_vda:
                if "sam_masks" in locals() and "sam_masks_dilate" in locals():
                    depths_mesh = (1 - sam_masks_dilate * sam_masks.squeeze()) * depths_t + (
                        sam_masks_dilate * sam_masks.squeeze() * depths_t.max()
                    )

            masks_rgb = masks.permute(0, 2, 3, 1).repeat(1, 1, 1, 3)
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
                for i in range(points_world.shape[0]):
                    _, _, warped_depth_pcd = run_render(
                        pcd=points_world[i : i + 1],
                        imgs=colors[i : i + 1],
                        masks=masks_rgb[i : i + 1],
                        H=H,
                        W=W,
                        c2ws=c2ws_target[i : i + 1],
                        K=intrinsic_target[i : i + 1],
                        num_views=1,
                        return_mask=True,
                        return_depth=True,
                        device=device,
                    )
                    warped_frame2, _, warped_depth_mesh = meshwarp.forward_warp(
                        colors[i : i + 1].permute(0, 3, 1, 2),
                        depths_mesh.unsqueeze(1)[i : i + 1],
                        torch.linalg.inv(Twcs_new[i : i + 1]),
                        w2cs_target[i : i + 1],
                        K_3x3[i : i + 1],
                        intrinsic_target[i : i + 1],
                    )
                    warped_images2.append(warped_frame2)

                    warped_mask_rgb, _, _ = meshwarp.forward_warp(
                        masks[i : i + 1].repeat(1, 3, 1, 1) * 2 - 1,
                        depths_mesh.unsqueeze(1)[i : i + 1],
                        torch.linalg.inv(Twcs_new[i : i + 1]),
                        w2cs_target[i : i + 1],
                        K_3x3[i : i + 1],
                        intrinsic_target[i : i + 1],
                    )
                    warped_images1.append(warped_mask_rgb)
                    masks1.append((warped_depth_pcd.to(warped_depth_mesh.device) <= warped_depth_mesh).to(torch.float32))

            cond_video = (torch.cat(warped_images2) + 1.0) / 2.0
            cond_video1 = (torch.cat(warped_images1) + 1.0) / 2.0
            cond_video1 = (cond_video1 >= 0.5).float()
            cond_masks1 = torch.cat(masks1)

            control_imgs = cond_video * cond_video1 * cond_masks1
            render_masks = cond_video1[:, 0:1] * cond_masks1
            control_imgs = control_imgs * 2.0 - 1.0

        elif render_method == "mesh":
            try:
                from utils.meshrenderex import MeshWarperEx
            except ModuleNotFoundError as exc:
                if getattr(exc, "name", None) == "pytorch3d":
                    raise ModuleNotFoundError(
                        "render_method='mesh' requires PyTorch3D, but it is not installed.\n"
                        "Install PyTorch3D (from source) following README.md, or switch to "
                        "`--render_method warp`."
                    ) from exc
                raise

            warped_images2 = []
            warped_images1 = []

            meshwarp = MeshWarperEx(resolution=(H, W), device=str(device))
            depths_mesh = depths_t
            if self.opts.advanced_render and self.opts.align_with_vda:
                if "sam_masks" in locals() and "sam_masks_dilate" in locals():
                    depths_mesh = (1 - sam_masks_dilate * sam_masks.squeeze()) * depths_t + (
                        sam_masks_dilate * sam_masks.squeeze() * depths_t.max()
                    )

            whitemesh = torch.ones_like(colors)
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
                for i in range(points_world.shape[0]):
                    warped_frame2, _, _ = meshwarp.forward_warp(
                        colors[i : i + 1].permute(0, 3, 1, 2),
                        depths_mesh.unsqueeze(1)[i : i + 1],
                        torch.linalg.inv(Twcs_new[i : i + 1]),
                        w2cs_target[i : i + 1],
                        K_3x3[i : i + 1],
                        intrinsic_target[i : i + 1],
                    )
                    warped_images2.append(warped_frame2)

                    warped_frame1, _, _ = meshwarp.forward_warp(
                        whitemesh[i : i + 1].permute(0, 3, 1, 2),
                        depths_mesh.unsqueeze(1)[i : i + 1],
                        torch.linalg.inv(Twcs_new[i : i + 1]),
                        w2cs_target[i : i + 1],
                        K_3x3[i : i + 1],
                        intrinsic_target[i : i + 1],
                    )
                    warped_images1.append((warped_frame1 > 0.0).float())

            cond_video = (torch.cat(warped_images2) + 1.0) / 2.0
            cond_masks = torch.cat(warped_images1)[:, 0:1]
            control_imgs = cond_video * 2.0 - 1.0
            render_masks = cond_masks

        else:
            if self.opts.advanced_render and self.opts.warp_with_occlusion:
                with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
                    # warp fg mask
                    points_world_input = points_world.clone()
                    colors_input = colors.clone()
                    normals_world_input = normals_world.clone()
                    masks_input = masks.clone()
                    for i in range(T):
                        masks_input[i] = (1-sam_masks[i]) * masks[i]
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
                        masks_input[i] = masks[i]
                    control_imgs_bg, render_masks_bg, render_depth_bg, render_depth_masks_bg = self.renderer.render(
                        c2ws=c2ws_target,
                        Ks=intrinsic_target,
                        points_world_from_img1=points_world_input,
                        colors_from_img1=colors_input,
                        mask_img1=masks_input,
                        normal_world_from_img1=None,
                        vis_threshold=getattr(self.opts, "vis_threshold", -0.1),
                    )
                    # 识别渲染过程中前景对后景的遮挡关系
                    print(inconsis_mask.shape)
                    save_video(inconsis_mask.permute(0,2,3,1).repeat(1, 1, 1, 3), os.path.join(self.opts.save_dir, 'mask_inconsis.mp4'), fps=self.opts.fps)

                    #del points_world_input, colors_input, normals_world_input, masks_input
                    # 将遮挡的背景反向warp到初始相机轨迹
                    # 计算points_world
                    intrinsic_target_inv = torch.linalg.inv(intrinsic_target)
                    depths_hw = render_depth_bg.view(T, H * W)
                    for i in range(T):
                        Ki_inv = intrinsic_target_inv[0]
                        rays = (Ki_inv @ uv1[i].T).T               # (HW,3)
                        Xi_cam = rays * depths_hw[i][:, None]      # (HW,3)

                        R_c2w = c2ws_target[i][:3, :3]
                        t_c2w = c2ws_target[i][:3, 3]
                        Xi_world = (R_c2w @ Xi_cam.T).T + t_c2w[None]
                        Xi_world = Xi_world.view(H, W, 3)

                        points_world_input[i] = Xi_world
                        colors_input[i] = control_imgs_bg[i].permute(1, 2, 0)
                        masks_input[i] = 1-inconsis_mask[i]

                    _, _, _, render_masks_occ = self.renderer.render(
                        c2ws=Twcs_new,
                        Ks=K_3x3,
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
                        masks_input[i] = masks_input[i] + (1-sam_masks[i])
                        points_world_input[i] = points_world[i]
                        colors_input[i] = colors[i]
                        normals_world_input[i] = normals_world[i]
                    masks_input[masks_input>0] = 1
                    masks_input *= masks
                    control_imgs, render_masks, _, _ = self.renderer.render(
                        c2ws=c2ws_target,
                        Ks=intrinsic_target,
                        points_world_from_img1=points_world_input,
                        colors_from_img1=colors_input,
                        mask_img1=masks_input,
                        normal_world_from_img1=normals_world_input,
                        vis_threshold=getattr(self.opts, "vis_threshold", -0.1),
                    )
            else:
                # auto: 6) Render
                with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
                    control_imgs, render_masks, _, _ = self.renderer.render(
                        c2ws=c2ws_target,
                        Ks=intrinsic_target,
                        points_world_from_img1=points_world,
                        colors_from_img1=colors,
                        mask_img1=masks,
                        normal_world_from_img1=normals_world,
                        vis_threshold=getattr(self.opts, "vis_threshold", -0.1),
                    )
        # Fit to diffusion resolution
        target_h, target_w = int(self.opts.height), int(self.opts.width)
        control_imgs = F.interpolate(control_imgs, size=[target_h, target_w], mode='bilinear', align_corners=False)
        render_masks = F.interpolate(render_masks.float(), size=[target_h, target_w], mode='nearest')

        # 7) Visualization & diffusion
        control_imgs[0:1] = (frames_torch[0:1]).permute(0, 3, 1, 2) * 2.0 - 1.0
        render_masks[0:1] = 1.0

        frames_vis = frames_torch
        save_video(frames_vis, os.path.join(self.opts.save_dir, 'input.mp4'), fps=self.opts.fps)

        render_results = einops.rearrange(control_imgs, "f c h w -> f h w c", f=T)
        view_masks = einops.rearrange(render_masks, "f c h w -> f h w c", f=T)
        view_masks = 1.0 - view_masks

        save_video((render_results + 1.0) / 2.0, os.path.join(self.opts.save_dir, 'render.mp4'), fps=self.opts.fps)
        save_video(view_masks.repeat(1, 1, 1, 3), os.path.join(self.opts.save_dir, 'mask.mp4'), fps=self.opts.fps)

        pil_mid = Image.fromarray((frames_np[T // 2] * 255).astype(np.uint8))
        prompt = self.get_caption(pil_mid)
        print(prompt)
        ref_video = (frames_torch[:10].permute(3, 0, 1, 2).unsqueeze(0) * 2.0 - 1.0)
        diffusion_results1 = self.run_diffusion(render_results, view_masks, prompt, ref_video=ref_video)
        diffusion_results = self.color_correct(diffusion_results1, (render_results + 1.0) / 2.0)
        save_video(diffusion_results1, os.path.join(self.opts.save_dir, 'diffusion_uncorrect.mp4'), fps=self.opts.fps)
        save_video(diffusion_results, os.path.join(self.opts.save_dir, 'diffusion_correct.mp4'), fps=self.opts.fps)

        tensor_left = frames_vis.to(device)
        tensor_right = diffusion_results.to(device)
        # UniView diffusion may adjust `num_frames` internally; keep visualization robust by aligning lengths.
        t_vis = min(int(tensor_left.shape[0]), int(tensor_right.shape[0]))
        tensor_left = tensor_left[:t_vis]
        tensor_right = tensor_right[:t_vis]
        interval = torch.ones(t_vis, self.opts.height, 30, 3, device=device)
        final_result = torch.cat((tensor_left, interval, tensor_right), dim=2)
        save_video(final_result, os.path.join(self.opts.save_dir, 'diffusion.mp4'), fps=self.opts.fps)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def color_correct(self, diffusion_results: torch.Tensor, render_results: torch.Tensor) -> torch.Tensor:
        gen_first = (diffusion_results[0].cpu().numpy() * 255).astype(np.float32)
        ref_first = (render_results[0].cpu().numpy() * 255).astype(np.float32)
        src_flat = gen_first.reshape(-1, 3)
        tgt_flat = ref_first.reshape(-1, 3)
        X = np.hstack([src_flat, np.ones((src_flat.shape[0], 1))])
        Y = tgt_flat
        W_, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        A = W_[:3].T
        b = W_[3]
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
    
    def save_images_folder(self, frames_np, folder_path):
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        os.makedirs(folder_path)
        for i in range(frames_np.shape[0]):
            image_cur = Image.fromarray((frames_np[i]*255).astype(np.uint8))
            image_cur.save(os.path.join(folder_path, f"{i:05d}.jpg"))

    def run_dynamic_view_gradio(self, i2v_input_video, i2v_stride, i2v_elevation, i2v_center_scale, i2v_pose, i2v_steps, i2v_seed, i2v_guidance_scale, vis_threshold):
        self._refresh_gradio_save_dir()

        self.opts.stride = i2v_stride
        self.opts.radius_scale = i2v_center_scale
        self.opts.image_dir = i2v_input_video
        self.opts.ddim_steps = i2v_steps
        try:
            self.opts.diffusion_guidance_scale = float(i2v_guidance_scale)
        except Exception:
            self.opts.diffusion_guidance_scale = getattr(self.opts, "diffusion_guidance_scale", 1.0)
        self.opts.elevation = float(i2v_elevation)

        pose_text = str(i2v_pose).strip()
        if pose_text.lower().startswith("swing"):
            self.opts.traj_type = "swing1"
            self.opts.d_theta = 0.0
            self.opts.d_phi = 0.0
            self.opts.x_offset = 0.0
            self.opts.y_offset = 0.0
            self.opts.z_offset = 0.0
            self.opts.d_r = getattr(self.opts, "d_r", 1.0)
        else:
            self.opts.traj_type = "custom"
            try:
                pose_vals = [float(x) for x in pose_text.replace(",", ";").split(";") if x.strip() != ""]
            except Exception:
                pose_vals = []
            self.opts.d_theta = getattr(self.opts, "d_theta", 0.0)
            self.opts.d_phi = getattr(self.opts, "d_phi", 0.0)
            self.opts.x_offset = getattr(self.opts, "x_offset", 0.0)
            self.opts.y_offset = getattr(self.opts, "y_offset", 0.0)
            self.opts.z_offset = getattr(self.opts, "z_offset", 0.0)
            self.opts.d_r = getattr(self.opts, "d_r", 1.0)
            if len(pose_vals) >= 2:
                self.opts.d_phi = pose_vals[0]
                self.opts.d_theta = pose_vals[1]
            if len(pose_vals) >= 3:
                self.opts.x_offset = pose_vals[2]
            if len(pose_vals) >= 4:
                self.opts.y_offset = pose_vals[3]
            if len(pose_vals) >= 5:
                self.opts.z_offset = pose_vals[4]

        self.opts.vis_threshold = float(vis_threshold)
        self.nvs_dynamic_view()
        gen_dir = os.path.join(self.opts.save_dir, "diffusion.mp4")
        return gen_dir
    
    def align_video_depth(self, frame_list, prompt_result, align_mask):
        '''
        prompt_result torch.tensor metric depth from stream3r
        '''
        self.update_momentum = 0.99 
        self.cache_scale_bias = None
        video_depth_model = VideoDepthAnythingDepthModel(model="vitl", device=self.opts.device)
        video_depth_result: torch.Tensor = unpack_optional(
                video_depth_model.estimate(DepthEstimationInput(video_frame_list=frame_list)).relative_inv_depth
            )
        depth_masks = self.reliable_depth_mask_range_batch(video_depth_result.unsqueeze(1).reciprocal(), window_size=5, ratio_thresh=0.1).squeeze()  # (T,1,H,W)

        video_inv_depth_total = []
        #for frame_idx, frame in pbar(enumerate(data_iterator), desc="Aligning depth"):
        for frame_idx, frame in enumerate(frame_list):
            video_depth_inv_depth = video_depth_result[frame_idx]
            video_depth_inv_depth = video_depth_result[frame_idx]
            mask_cur = align_mask[frame_idx].cpu().numpy()
            mask_cur = cv2.erode(mask_cur, np.ones((5, 5)), iterations=2).astype(np.uint8)
            mask_cur = torch.from_numpy(mask_cur).to(align_mask.device)

            sparse_mask = mask_cur * depth_masks[frame_idx] * (video_depth_inv_depth > 1e-3)
            keep_ratio = 0.25                                 # 保留 25%，可调
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
                scale, bias = self.cache_scale_bias
            
            if self.cache_scale_bias is None:
                self.cache_scale_bias = (scale, bias)
            scale = self.cache_scale_bias[0] * self.update_momentum + scale * (1 - self.update_momentum)
            bias = self.cache_scale_bias[1] * self.update_momentum + bias * (1 - self.update_momentum)
            self.cache_scale_bias = (scale, bias)

            video_inv_depth = video_depth_inv_depth * scale + bias
            video_inv_depth[video_inv_depth < 1e-3] = 1e-3
            video_inv_depth = video_inv_depth.reciprocal()
            video_inv_depth_total.append(video_inv_depth)
        
        self._safe_move_module(getattr(video_depth_model, "model", None), "cpu")
        del video_depth_model
        # qing缓存
        torch.cuda.empty_cache()

        video_inv_depth = torch.stack(video_inv_depth_total)
        return video_inv_depth
