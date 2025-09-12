import sys
import os
from pathlib import Path
from typing import Tuple, List

# Make STream3R importable
sys.path.insert(0, str(Path(__file__).parent / "extern/STream3R"))

# single view packages (kept the same as demo_dynamic_vipe for downstream rendering flow)
from carvekit.ml.wrap.tracer_b7 import TracerUniversalB7
from diffusers.utils import export_to_video

from src.utils import traj_map, points_padding, np_points_padding, set_initial_camera, build_cameras
from datetime import datetime
from extern.moge.model.v2 import MoGeModel  # Let's try MoGe-2

from utils.warp_utils import *  # read_video_frames, save_video, etc.
import hydra
from src.imagesplatrender import ImageSplattingRenderer

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
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R

# diffusion packages
from diffusers import AutoencoderKLWan
from vace_diffusers.pipeline_wan_vace import WanVACEPipeline
from vace_diffusers.vace_transformer import WanVACETransformer3DModel
from diffusers import FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler
from transformers import AutoTokenizer, UMT5EncoderModel

# blip packages
from transformers import AutoProcessor, Blip2ForConditionalGeneration

# STream3R
from stream3r.models.stream3r import STream3R
from stream3r.models.components.utils.pose_enc import pose_encoding_to_extri_intri
from stream3r.models.components.utils.geometry import closed_form_inverse_se3
from utils.tensor_stream import _preprocess_frames_for_stream3r


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
        self.setup_stream3r()
        self.setup_moge()
        # self.setup_diffusion()

        self.seg_net = TracerUniversalB7(device='cuda', batch_size=1, model_path=self.opts.segnet_path).eval()
        self.caption_processor = AutoProcessor.from_pretrained(opts.blip_path)
        self.captioner = Blip2ForConditionalGeneration.from_pretrained(opts.blip_path, torch_dtype=torch.float16).to(opts.device)
        self.renderer = ImageSplattingRenderer(resolution=(opts.height, opts.width), device=opts.device)

    def setup_stream3r(self):
        # Load pretrained STream3R once
        self.stream3r = STream3R.from_pretrained("yslan/STream3R").to(self.opts.device).eval()

    def setup_moge(self):
        self.depth_model = MoGeModel.from_pretrained(self.opts.moge_path).to(self.opts.device).eval()

    def setup_diffusion(self):
        transformer = WanVACETransformer3DModel.from_pretrained(self.opts.transformer_path, torch_dtype=self.opts.weight_dtype)
        vae = AutoencoderKLWan.from_pretrained(self.opts.model_name, subfolder="vae", torch_dtype=torch.float32)

        tokenizer = AutoTokenizer.from_pretrained(os.path.join(self.opts.model_name, "tokenizer"))
        text_encoder = UMT5EncoderModel.from_pretrained(
            os.path.join(self.opts.model_name, "text_encoder"), low_cpu_mem_usage=True, torch_dtype=self.opts.weight_dtype
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
        self.pipeline.to(self.opts.device)
        if getattr(self.opts, "lora_path", None):
            self.pipeline.load_lora_weights(self.opts.lora_path, adapter_name="causvid_lora")
            self.pipeline.set_adapters(["causvid_lora"], adapter_weights=[0.95])
            self.pipeline.fuse_lora()

    def nvs_single_view(self, image=None):
        if image is None:
            image = Image.open(self.opts.image_dir).convert("RGB")
            image = ImageOps.exif_transpose(image)
            image = image.resize((self.opts.width, self.opts.height), Image.Resampling.BICUBIC)

        validation_image = ToTensor()(image)[None]  # [1,c,h,w], 0~1
        depth_image = validation_image[0].to(self.opts.device)
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

        points3d = points3d.cpu().numpy()
        colors = colors.cpu().numpy()
        normal = normal.cpu().numpy()

        # inference foreground mask
        with torch.no_grad():
            origin_w_, origin_h_ = image.size
            image_pil = image.resize((512, 512))
            fg_mask = self.seg_net([image_pil])[0]
            fg_mask = fg_mask.resize((origin_w_, origin_h_))

        fg_mask = np.array(fg_mask)
        fg_mask = fg_mask > 127.5
        fg_mask = torch.tensor(fg_mask)
        if fg_mask.float().mean() < 0.05:
            fg_mask[...] = True
        # 用前景中心深度作为点云世界坐标系的原点
        depth_avg = torch.median(depth[0, 0, fg_mask]).item()
        w2c_0, c2w_0 = set_initial_camera(self.opts.elevation, depth_avg)

        # 转为世界坐标
        points3d = (c2w_0.numpy()[:3] @ np_points_padding(points3d).T).T
        normal = (c2w_0[:3, :3] @ normal.T).T

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

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            control_imgs, render_masks = self.renderer.render(
                c2ws=c2ws,
                Ks=intrinsic,
                points_world_from_img1=torch.from_numpy(points3d)
                .reshape(self.opts.height, self.opts.width, 3)[None]
                .repeat(self.opts.video_length, 1, 1, 1),
                colors_from_img1=torch.from_numpy((colors / 255.0) * 2.0 - 1.0)
                .reshape(self.opts.height, self.opts.width, 3)[None]
                .repeat(self.opts.video_length, 1, 1, 1),
                mask_img1=masks.reshape(1, 1, self.opts.height, self.opts.width)
                .repeat(self.opts.video_length, 1, 1, 1)
                .float(),
                normal_world_from_img1=normal.reshape(self.opts.height, self.opts.width, 3)[None].repeat(
                    self.opts.video_length, 1, 1, 1
                ),
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

        diffusion_results1 = self.run_diffusion(render_results, view_masks, prompt)
        diffusion_results2 = self.color_correct(diffusion_results1, (render_results + 1.0) / 2.0)
        save_video(diffusion_results1, os.path.join(self.opts.save_dir, "diffusion_uncorrect.mp4"))
        save_video(diffusion_results2, os.path.join(self.opts.save_dir, "diffusion_correct.mp4"))
        return diffusion_results1

    def run_single_view_gradio(self, i2v_input_image, i2v_elevation, i2v_center_scale, i2v_pose, i2v_steps, i2v_seed, i2v_guidance_scale, vis_threshold):
        if not getattr(self.opts, "save_dir", None):
            prefix = datetime.now().strftime("%Y%m%d_%H%M")
            self.opts.save_dir = f"./output/gradio/{prefix}"
        os.makedirs(self.opts.save_dir, exist_ok=True)

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
        orig_width, orig_height = image.size

        if orig_width >= orig_height:
            new_width = 1024
            new_height = int(orig_height * (1024 / orig_width))
        else:
            new_height = 1024
            new_width = int(orig_width * (1024 / orig_height))
        image = image.resize((new_width, new_height), Image.BICUBIC)
        image = ImageOps.fit(image, (832, 480), method=Image.BICUBIC, centering=(0.5, 0.5))
        final_w, final_h = 832, 480

        self.opts.width = final_w
        self.opts.height = final_h
        print(image.size)
        print(f"Image resized to: {final_w}x{final_h}")

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

    def run_diffusion(self, cond_video, cond_masks, prompt):
        cond_video = cond_video.permute(3, 0, 1, 2).unsqueeze(0)  # (1, T, H, W, C)
        cond_masks = cond_masks.permute(3, 0, 1, 2).unsqueeze(0)  # (1, T, H, W, C)
        with torch.no_grad():
            steps = int(getattr(self.opts, "ddim_steps", 8))
            sample = self.pipeline(
                video=cond_video,
                mask=cond_masks,
                prompt=prompt,
                negative_prompt=self.opts.negative_prompt,
                height=self.opts.height,
                width=self.opts.width,
                num_frames=self.opts.video_length,
                num_inference_steps=steps,
                guidance_scale=float(getattr(self.opts, "diffusion_guidance_scale", 1.0)),
                generator=torch.Generator().manual_seed(42),
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

    def nvs_dynamic_view(self):
        device = self.opts.device
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
        assert T == self.opts.video_length, "读取到的帧数与期望不一致"

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

        # Depth ready for downstream (no VDA alignment)
            # Convert to c2w (cam->world) and align to our world using first frame and initial camera
        Twcs_stream = closed_form_inverse_se3(extri_34)  # (T,4,4)

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

        # 6) Render
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            control_imgs, render_masks = self.renderer.render(
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
        diffusion_results1 = self.run_diffusion(render_results, view_masks, prompt)
        diffusion_results = self.color_correct(diffusion_results1, (render_results + 1.0) / 2.0)
        save_video(diffusion_results1, os.path.join(self.opts.save_dir, 'diffusion_uncorrect.mp4'))
        save_video(diffusion_results, os.path.join(self.opts.save_dir, 'diffusion_correct.mp4'))

        tensor_left = frames_vis.to(device)
        tensor_right = diffusion_results.to(device)
        interval = torch.ones(T, self.opts.height, 30, 3, device=device)
        final_result = torch.cat((tensor_left, interval, tensor_right), dim=2)
        save_video(final_result, os.path.join(self.opts.save_dir, 'diffusion.mp4'), fps=self.opts.fps)

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

    def run_dynamic_view_gradio(self, i2v_input_video, i2v_stride, i2v_elevation, i2v_center_scale, i2v_pose, i2v_steps, i2v_seed, i2v_guidance_scale, vis_threshold):
        if not getattr(self.opts, "save_dir", None):
            prefix = datetime.now().strftime("%Y%m%d_%H%M")
            self.opts.save_dir = f'./output/gradio/{prefix}'
        os.makedirs(self.opts.save_dir, exist_ok=True)

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
