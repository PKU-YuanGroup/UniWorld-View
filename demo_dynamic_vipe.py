import sys
# single view packages
from carvekit.ml.wrap.tracer_b7 import TracerUniversalB7
from diffusers.utils import export_to_video
# from pytorch3d.renderer import PointsRasterizationSettings
# from src.pointcloud import point_rendering
from src.utils import traj_map, points_padding, np_points_padding, set_initial_camera, build_cameras
from datetime import datetime
from extern.moge.model.v2 import MoGeModel # Let's try MoGe-2

# # sparse view packages
# import sys
# sys.path.append('./extern/dust3r')
# from dust3r.inference import inference, load_model
# from dust3r.utils.image import load_images
# from dust3r.image_pairs import make_pairs
# from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
# from dust3r.utils.device import to_numpy
# import pytorch3d
# from pytorch3d.structures import Pointclouds
# from utils.pvd_utils import *
# from omegaconf import OmegaConf
# from torchvision.utils import save_image

# dynamic view packages
# from extern.DepthCrafter.infer import DepthCrafterDemo
from utils.warp_utils import *
import hydra
from src.imagesplatrender import ImageSplattingRenderer
from pathlib import Path


# other packages
import json
import shutil
import os
import einops
import warnings
import trimesh
import torch
import numpy as np
import torchvision
import copy
import cv2  
import glob
from PIL import Image
from torchvision.utils import save_image
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from torchvision.transforms import ToTensor, ToPILImage

# diffusion packages
from diffusers import AutoencoderKLWan
from vace_diffusers.pipeline_wan_vace import WanVACEPipeline
from vace_diffusers.vace_transformer import WanVACETransformer3DModel
from diffusers import FlowMatchEulerDiscreteScheduler,UniPCMultistepScheduler
from transformers import AutoTokenizer, UMT5EncoderModel


# blip packages
from transformers import AutoProcessor, Blip2ForConditionalGeneration

# Prefer local VIPE copy under extern_vipe
sys.path.insert(0, str(Path(__file__).parent / "extern/vipe"))
from vipe import get_config_path, make_pipeline  
from vipe.streams.tensor_stream import TensorVideoStream 

class UniScene:
    def __init__(self, opts, gradio = False):
        self.opts = opts
        self.device = opts.device
        self.setup_vipe()
        self.setup_moge()
        self.setup_diffusion()
        # self.setup_dust3r()
        self.caption_processor = AutoProcessor.from_pretrained(opts.blip_path)
        self.captioner = Blip2ForConditionalGeneration.from_pretrained(opts.blip_path, torch_dtype=torch.float16).to(opts.device)
        self.renderer = ImageSplattingRenderer(resolution=(opts.height, opts.width), device=opts.device)
        
        # initialize ref images, pcd
        # if not gradio:
        #     if os.path.isfile(self.opts.image_dir):
        #         self.images, self.img_ori = self.load_initial_images(image_dir=self.opts.image_dir)
        #         self.run_dust3r(input_images=self.images)
        #     elif os.path.isdir(self.opts.image_dir):
        #         self.images, self.img_ori = self.load_initial_dir(image_dir=self.opts.image_dir)
        #         self.run_dust3r(input_images=self.images, clean_pc = True)    
        #     else:
        #         print(f"{self.opts.image_dir} doesn't exist")   

    # def run_dust3r(self, input_images,clean_pc = False):
    #     pairs = make_pairs(input_images, scene_graph='complete', prefilter=None, symmetrize=True)
    #     output = inference(pairs, self.dust3r, self.device, batch_size=self.opts.batch_size)

    #     mode = GlobalAlignerMode.PointCloudOptimizer #if len(self.images) > 2 else GlobalAlignerMode.PairViewer
    #     scene = global_aligner(output, device=self.device, mode=mode)
    #     if mode == GlobalAlignerMode.PointCloudOptimizer:
    #         loss = scene.compute_global_alignment(init='mst', niter=self.opts.niter, schedule=self.opts.schedule, lr=self.opts.lr)

    #     if clean_pc:
    #         self.scene = scene.clean_pointcloud()
    #     else:
    #         self.scene = scene

    # def render_pcd(self,pts3d,imgs,masks,views,renderer,device,nbv=False):
        
    #     imgs = to_numpy(imgs)
    #     pts3d = to_numpy(pts3d)

    #     if masks == None:
    #         pts = torch.from_numpy(np.concatenate([p for p in pts3d])).view(-1, 3).to(device)
    #         col = torch.from_numpy(np.concatenate([p for p in imgs])).view(-1, 3).to(device)
    #     else:
    #         # masks = to_numpy(masks)
    #         pts = torch.from_numpy(np.concatenate([p[m] for p, m in zip(pts3d, masks)])).to(device)
    #         col = torch.from_numpy(np.concatenate([p[m] for p, m in zip(imgs, masks)])).to(device)
        
    #     point_cloud = Pointclouds(points=[pts], features=[col]).extend(views)
    #     images = renderer(point_cloud)

    #     if nbv:
    #         color_mask = torch.ones(col.shape).to(device)
    #         point_cloud_mask = Pointclouds(points=[pts],features=[color_mask]).extend(views)
    #         view_masks = renderer(point_cloud_mask).mean(dim=-1).unsqueeze(-1)
    #         view_masks = (view_masks >= 0.1).float()
    #     else: 
    #         view_masks = None

    #     return images, view_masks
    
    # def run_render(self, pcd, imgs,masks, H, W, camera_traj,num_views,nbv=False):
    #     render_setup = setup_renderer(camera_traj, image_size=(H,W))
    #     renderer = render_setup['renderer']
    #     render_results, viewmask = self.render_pcd(pcd, imgs, masks, num_views,renderer,self.device,nbv)
    #     return render_results, viewmask

    def run_moge(self, depth_image):
        with torch.no_grad():
            output = self.depth_model.infer(depth_image)
            depth = output["depth"]  # Depth in [m]. 
            normal = output["normal"]  # Normal in [h,w,3]
            moge_mask = output["mask"]  
            depth = torch.where(
                moge_mask == 0,
                torch.tensor(1000.0, device=self.opts.device),
                depth
            )
            background_normal = torch.tensor([0.0, 0.0, -1.0], device=normal.device)

            normal = torch.where(
                moge_mask[..., None] == 0,  # shape [H, W, 1]
                background_normal[None, None, :],  # broadcast to [H, W, 3]
                normal
            ).view(-1,3)
            # points3d = output["points"].view(-1,3).clone()  # Points in camera coordinate, [h*w, 3]
            masks = self.reliable_depth_mask_range_batch(depth.unsqueeze(0), window_size=5, ratio_thresh=0.1).view(-1).bool()  # Masks in [h*w], True for valid points
            K_normalized = output["intrinsics"]
            # Convert the normalized intrinsics to pixel coordinates
            K = K_normalized.clone()
            K[0, 0] *= self.opts.width
            K[1, 1] *= self.opts.height
            K[0, 2] *= self.opts.width
            K[1, 2] *= self.opts.height
            depth = depth[None, None]
            K_inv = K.inverse()
            intrinsic = K[None].repeat(self.opts.video_length, 1, 1)

            return  depth, masks, normal, intrinsic, K, K_inv     

    
    def run_diffusion(self, cond_video, cond_masks,prompt):
        """
        cond_video: torch.Tensor, shape (H, W, C, T)
        cond_masks: torch.Tensor, shape (H, W, C, T), binary mask
        prompt: str
        """

        # torch.Size([1, 3, 81, 480, 832]) torch.Size([1, 1, 81, 480, 832]) PIL.Image
        # === 输入处理 ===
        cond_video = cond_video.permute(3, 0, 1, 2).unsqueeze(0)  # (1, T, H, W, C)
        cond_masks = cond_masks.permute(3, 0, 1, 2).unsqueeze(0)   # (1, T, H, W, C)
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
                guidance_scale=1,
                generator=torch.Generator().manual_seed(42),
            ).frames[0]

        return torch.from_numpy(sample)  # → shape (F, H, W, C)


    def nvs_single_view(self, image = None):
        if image is None:
            image = Image.open(self.opts.image_dir).convert("RGB")
            image = ImageOps.exif_transpose(image)
            image = image.resize((self.opts.width, self.opts.height), Image.Resampling.BICUBIC)
        
        validation_image = ToTensor()(image)[None]  # [1,c,h,w], 0~1
        depth_image = validation_image[0].to(self.opts.device)
        depth, masks, normal, intrinsic, K, K_inv   = self.run_moge(depth_image)

        # get pointcloud
        points2d = torch.stack(torch.meshgrid(torch.arange(self.opts.width, dtype=torch.float32),
                                            torch.arange(self.opts.height, dtype=torch.float32), indexing="xy"), -1).to(self.opts.device)  # [h,w,2]
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
        # fg_mask的作用是分割出前景, 用前景的中心深度作为点云世界坐标系的原点
        depth_avg = torch.median(depth[0, 0, fg_mask]).item()
        w2c_0, c2w_0 = set_initial_camera(self.opts.elevation, depth_avg)

        # convert points3d to the world coordinate
        points3d = (c2w_0.numpy()[:3] @ np_points_padding(points3d).T).T

        # convert normal to the world coordinate
        normal = (c2w_0[:3, :3] @ normal.T).T  # 结果也是 [N, 3]

        # pcd = trimesh.PointCloud(vertices=points3d, colors=colors)
        # _ = pcd.export(f"{self.opts.save_dir}/pcd.ply")

        # == motion definition ==
        if self.opts.traj_type == "custom":
            cam_traj, x_offset, y_offset, z_offset, d_theta, d_phi, d_r = \
                "free", self.opts.x_offset, self.opts.y_offset, self.opts.z_offset, self.opts.d_theta, self.opts.d_phi, self.opts.d_r
        else:
            cam_traj, x_offset, y_offset, z_offset, d_theta, d_phi, d_r = traj_map(self.opts.traj_type)
        
        w2cs, c2ws, intrinsic = build_cameras(cam_traj=cam_traj,
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
                                            z_offset=z_offset)


        # render_results, viewmask = self.run_render([pcd[-1]], [imgs[-1]],masks, H, W, camera_traj,num_views, nbv=True)    
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            control_imgs, render_masks = self.renderer.render(
                                    c2ws=c2ws,
                                    Ks=intrinsic,
                                    points_world_from_img1=torch.from_numpy(points3d).reshape(self.opts.height, self.opts.width, 3)[None].repeat(self.opts.video_length,1,1,1),
                                    colors_from_img1=torch.from_numpy((colors/255.)*2.-1.).reshape(self.opts.height, self.opts.width, 3)[None].repeat(self.opts.video_length,1,1,1),
                                    mask_img1=masks.reshape(1,1,self.opts.height,self.opts.width).repeat(self.opts.video_length,1,1,1).float(),  # Optional mask
                                    normal_world_from_img1 = normal.reshape(self.opts.height, self.opts.width, 3)[None].repeat(self.opts.video_length,1,1,1),  # Optional normals
                                )

        control_imgs[0:1] = validation_image*2.-1.
        render_masks [0:1] = 1.

        render_results = einops.rearrange(control_imgs, "f c h w -> f h w c", f=self.opts.video_length)
        view_masks = einops.rearrange(render_masks, "f c h w -> f h w c", f=self.opts.video_length)
        view_masks = 1.-view_masks
        
        # render_results, view_masks = self.clean_points((render_results+1.)/2.,1-view_masks)  
        render_results = (render_results+1.)/2.
        save_video(render_results, os.path.join(self.opts.save_dir, f'render.mp4'))
        save_video(view_masks.repeat(1,1,1,3), os.path.join(self.opts.save_dir, f'mask.mp4'))

        render_results = render_results*2.-1.

        prompt = self.get_caption(image)
        print(prompt)
    
        diffusion_results = self.run_diffusion(render_results, view_masks, prompt) # torch.Size([49, 384, 672, 3])
        diffusion_results = self.color_correct(diffusion_results, (render_results+1.)/2.) # color correction
        save_video(diffusion_results, os.path.join(self.opts.save_dir, f'diffusion.mp4'))
        # save_video(diffusion_results2, os.path.join(self.opts.save_dir, f'diffusion_correct.mp4'))
        # frame_dir = os.path.join(self.opts.save_dir,'frames')
        # os.makedirs(frame_dir,exist_ok=True)

        # for i in range(diffusion_results.shape[0]):
        #     # 获取当前帧
        #     frame = diffusion_results[i].permute(2, 0, 1)  # 从 [H, W, C] 转为 [C, H, W]
        #     frame = frame * 255.0
        #     frame = frame.clamp(0, 255).byte()  # 限制范围并转换为 byte 类型

        #     # 转换为 PIL 图像
        #     pil_image = Image.fromarray(frame.permute(1, 2, 0).numpy())  # 从 [C, H, W] 转为 [H, W, C]
            
        #     # 保存为 PNG 文件
        #     pil_image.save(os.path.join(frame_dir, f"frame_{i}.png"))

        # viz = True
        # if viz:
        #     tensor_left = diffusion_results[0:1].repeat(49,1,1,1).to(self.device)
        #     tensor_right = diffusion_results.to(self.device)
        #     interval = torch.ones(49, 384, 30, 3).to(self.device)
        #     result = torch.cat((tensor_left, interval, tensor_right), dim=2)
        #     # result_reverse = torch.flip(result, dims=[0])
        #     # final_result = torch.cat((result, result_reverse[1:,:,:,:]), dim=0)
        #     save_video(result, os.path.join(self.opts.save_dir,f'viz{itr}.mp4'), fps=8)

        return diffusion_results

    def run_single_view_gradio(self,i2v_input_image, i2v_elevation, i2v_center_scale, i2v_pose,i2v_steps, i2v_seed):
        
        # 若外部已设置 save_dir（app_dynamic.py 启动时已设置），则沿用，不再新建二级目录
        if not getattr(self.opts, "save_dir", None):
            prefix = datetime.now().strftime("%Y%m%d_%H%M")
            self.opts.save_dir = f'./output/gradio/{prefix}'
        os.makedirs(self.opts.save_dir, exist_ok=True)
        
        # fix seed
        self.opts.ddim_steps = i2v_steps
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
            vals = [float(x) for x in pose_text.replace(",",";").split(";") if x.strip() != ""]
            while len(vals) < 5:
                vals.append(0.0)
            self.opts.d_phi, self.opts.d_theta, self.opts.x_offset, self.opts.y_offset, self.opts.z_offset = vals[:5]

        image = Image.fromarray(i2v_input_image).convert("RGB")
        w, h = image.size

        # if orig_width >= orig_height:
        #     new_width = 1024
        #     new_height = int(orig_height * (1024 / orig_width))
        # else:
        #     new_height = 1024
        #     new_width = int(orig_width * (1024 / orig_height))
        # image = image.resize((new_width, new_height), Image.BICUBIC)        
        # image = ImageOps.fit(image, (832, 480), method=Image.BICUBIC, centering=(0.5, 0.5))
  

        # # 第一步：长边缩放到 832，短边等比缩放
        if w > h:
            new_w = 832
            new_h = int(h * (832 / w))
        else:
            new_h = 832
            new_w = int(w * (832 / h))

        # 第二步：将 w, h 调整为最近的能被16整除的尺寸（向下取整）
        final_w = (new_w // 16) * 16
        final_h = (new_h // 16) * 16

        image = image.resize((final_w, final_h), Image.BICUBIC)    
        self.opts.width = final_w
        self.opts.height = final_h    
        print(image.size)
        print(f"Image resized to: {final_w}x{final_h}")
        
        self.nvs_single_view(image)
        # traj_dir = os.path.join(self.opts.save_dir, "viz_traj.mp4")
        gen_dir = os.path.join(self.opts.save_dir, "diffusion.mp4")
        return gen_dir #,traj_dir,

    # def nvs_sparse_view(self):
    #     diffusion_results = []
    #     image_pairs, gt_pairs = self.load_initial_dir(image_dir=self.opts.image_dir) 

    #     for i in range(len(image_pairs)):
    #         images = image_pairs[i]
    #         img_ori = gt_pairs[i]
    #         print(f'Generating {i} clips\n')
    #         self.run_dust3r(input_images=images, clean_pc = True)   
    #         c2ws = self.scene.get_im_poses().detach()
    #         principal_points = self.scene.get_principal_points().detach()
    #         focals = self.scene.get_focals().detach()
    #         shape = images[0]['true_shape']
    #         H, W = int(shape[0][0]), int(shape[0][1])
    #         pcd = [j.detach() for j in self.scene.get_pts3d(clip_thred=self.opts.dpt_trd)] # a list of points of size whc
    #         masks = None
    #         mask_pc = False
    #         imgs = np.array(self.scene.imgs)
    #         # [2,1] [2,2]
    #         camera_traj,num_views = generate_traj_interp(c2ws, H, W, focals, principal_points, self.opts.video_length, self.device)
    #         render_results, viewmask = self.run_render(pcd, imgs,masks, H, W, camera_traj,num_views,nbv=True)
    #         render_results, viewmask = self.clean_points(render_results,viewmask)           
    #         render_results = F.interpolate(render_results.permute(0,3,1,2), size=(576, 1024), mode='bilinear', align_corners=False).permute(0,2,3,1)
    #         render_results[0] = img_ori[0]
    #         render_results[-1] = img_ori[-1]
    #         render_results = F.interpolate(render_results.permute(0,3,1,2), size=(self.opts.height, self.opts.width), mode='bilinear', align_corners=False).permute(0,2,3,1)
    #         viewmask = F.interpolate(viewmask.permute(0,3,1,2), size=(self.opts.height, self.opts.width), mode='nearest').permute(0,2,3,1)   
    #         save_video(render_results, os.path.join(self.opts.save_dir, f'render{i}.mp4'))
    #         save_video(viewmask.repeat(1,1,1,3), os.path.join(self.opts.save_dir, f'mask{i}.mp4'))
    #         # save_pointcloud_with_normals(imgs, pcd, msk=masks, save_path=os.path.join(self.opts.save_dir, f'pcd.ply') , mask_pc=mask_pc, reduce_pc=False)

    #         # if self.opts.prompt != '':
    #         #     prompt = self.opts.prompt + '. The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic.'
    #         # else:
    #         prompt = self.get_caption(img_ori[0].cpu().numpy())
    #         print(prompt)
    #         render_results = render_results*2.-1.
    #         viewmask = 1.-viewmask
    #         diffusion_results_tmp = self.run_diffusion(render_results,viewmask,prompt)
    #         diffusion_results_tmp = self.color_correct(diffusion_results_tmp, (render_results+1.)/2.) # color correction
    #         save_video(diffusion_results_tmp, os.path.join(self.opts.save_dir, f'diffusion{i}.mp4'))
    #         diffusion_results.append(diffusion_results_tmp)

    #     diffusion_results = torch.cat(diffusion_results)
    #     save_video(diffusion_results, os.path.join(self.opts.save_dir, f'diffusion.mp4'), fps=30)
    #     print(f'Finish!\n')
    #     # torch.Size([25, 576, 1024, 3])
    #     return diffusion_results

    # def run_sparse_view_gradio(self,i2v_input_image, i2v_steps, i2v_seed):
    #     # fix seed
    #     prefix = datetime.now().strftime("%Y%m%d_%H%M")
    #     target_dir_images = os.path.join(self.opts.save_dir, f"{prefix}_images")
    #     os.makedirs(target_dir_images, exist_ok=True)

    #     for file_data in i2v_input_image:
    #         # 安全地提取路径：支持 dict / TemporaryFileWrapper / str
    #         if isinstance(file_data, dict) and "name" in file_data:
    #             file_path = file_data["name"]
    #         elif hasattr(file_data, "name"):  # e.g., TemporaryFileWrapper
    #             file_path = file_data.name
    #         elif isinstance(file_data, str):
    #             file_path = file_data
    #         else:
    #             raise TypeError(f"不支持的上传类型: {type(file_data)}")

    #         # 复制到目标目录
    #         dst_path = os.path.join(target_dir_images, os.path.basename(file_path))
    #         shutil.copy(file_path, dst_path)

    #     self.opts.image_dir = target_dir_images
    #     self.opts.ddim_steps = i2v_steps
    #     self.nvs_sparse_view()
    #     # traj_dir = os.path.join(self.opts.save_dir, "viz_traj.mp4")
    #     gen_dir = os.path.join(self.opts.save_dir, "diffusion.mp4")
    #     return gen_dir #,traj_dir,


    # def nvs_dynamic_view(self):

    #     frames = read_video_frames(
    #         self.opts.image_dir, self.opts.video_length, self.opts.stride, self.opts.max_res
    #     )

    #     image_array = (frames[self.opts.video_length // 2]* 255).astype(np.uint8)
    #     pil_image = Image.fromarray(image_array)

    #     # 使用 VIPE 对已加载的 frames（numpy float32, [0,1], 形状 [T,H,W,3]）做深度估计
    #     # 改为传入内存视频张量，避免二次读取磁盘（参考 test_tensor_stream.py 的 TensorVideoStream 用法）
    #     video_tensor = torch.from_numpy(frames)  # (T,H,W,3) float32 in [0,1]
    #     s = self.vipe.run(TensorVideoStream(video_tensor, fps=self.opts.fps, name="nvs_dynamic")).output_streams[0]
    #     vipe_depths = torch.stack([f.metric_depth for f in s])  # [T, Hvi, Wvi]
    #     # Align depth to the frame size used downstream (576x1024 by default in read_video_frames)
    #     T, Hf, Wf = vipe_depths.shape[0], frames.shape[1], frames.shape[2]
    #     depths = vipe_depths.unsqueeze(1).to(self.opts.device)  # [T,1,Hvi,Wvi]
    #     depths = F.interpolate(depths, size=(Hf, Wf), mode="bilinear", align_corners=False)
    #     frames = (
    #         torch.from_numpy(frames).permute(0, 3, 1, 2).to(self.opts.device) * 2.0 - 1.0
    #     )  # 49 576 1024 3 -> 49 3 576 1024, [-1,1]
        
    #     assert frames.shape[0] == self.opts.video_length
    #     pose_s, pose_t, K = self.get_poses(self.opts, depths, num_frames=self.opts.video_length)
        
    #     depth_masks = self.reliable_depth_mask_range_batch(depths, window_size=5, ratio_thresh=0.1)  # Masks in [h*w], True for valid points

    #     warped_images = []
    #     masks = []
    #     for i in tqdm(range(self.opts.video_length)):
    #         warped_frame2, mask2, warped_depth2, flow12 = self.funwarp.forward_warp(
    #             frames[i : i + 1],
    #             depth_masks[i : i + 1],
    #             depths[i : i + 1],
    #             pose_s[i : i + 1],
    #             pose_t[i : i + 1],
    #             K[i : i + 1],
    #             None,
    #             self.opts.mask,
    #             twice=False,
    #         )
    #         warped_images.append(warped_frame2)
    #         masks.append(mask2)

    #     cond_video = (torch.cat(warped_images) + 1.0) / 2.0
    #     cond_masks = torch.cat(masks)

    #     frames = F.interpolate(
    #         frames, size=[self.opts.height, self.opts.width], mode='bilinear', align_corners=False
    #     )
    #     cond_video = F.interpolate(
    #         cond_video, size=[self.opts.height, self.opts.width], mode='bilinear', align_corners=False
    #     )
    #     cond_masks = F.interpolate(cond_masks, size=[self.opts.height, self.opts.width], mode='nearest')
        
    #     cond_video[0] = (frames[0]+1.)/2.
    #     cond_masks[0] = torch.ones_like(cond_masks[0])

    #     frames = (frames.permute(0, 2, 3, 1) + 1.0) / 2.0
    #     save_video(
    #         frames,
    #         os.path.join(self.opts.save_dir, 'input.mp4'),
    #         fps=self.opts.fps,
    #     )

    #     render_results = cond_video.permute(0, 2, 3, 1)
    #     save_video(
    #         render_results,
    #         os.path.join(self.opts.save_dir, 'render.mp4'),
    #         fps=self.opts.fps,
    #     )

    #     view_masks = cond_masks.permute(0, 2, 3, 1)

    #     render_results = render_results*2.-1.
    #     view_masks = 1.-view_masks

    #     save_video(
    #         view_masks.repeat(1, 1, 1, 3),
    #         os.path.join(self.opts.save_dir, 'mask.mp4'),
    #         fps=self.opts.fps,
    #     )

    #     prompt = self.get_caption(pil_image)
    #     print(prompt)
    #     diffusion_results1 = self.run_diffusion(render_results, view_masks, prompt) # torch.Size([49, 384, 672, 3])
    #     diffusion_results = self.color_correct(diffusion_results1, (render_results+1.)/2.) # color correction
    #     save_video(diffusion_results1, os.path.join(self.opts.save_dir, f'diffusion_uncorrect.mp4'))
    #     save_video(diffusion_results, os.path.join(self.opts.save_dir, f'diffusion_correct.mp4'))
        
    #     viz = True
    #     if viz:
    #         tensor_left = frames.to(self.opts.device)
    #         tensor_right = diffusion_results.to(self.opts.device)
    #         interval = torch.ones(81, self.opts.height, 30, 3).to(self.opts.device)
    #         result = torch.cat((tensor_left, interval, tensor_right), dim=2)
    #         final_result = result
    #         # result_reverse = torch.flip(result, dims=[0])
    #         # final_result = torch.cat((result, result_reverse[1:, :, :, :]), dim=0)
    #         save_video(
    #             final_result,
    #             os.path.join(self.opts.save_dir, 'diffusion.mp4'),
    #             fps=self.opts.fps,
    #         )
    

    def nvs_dynamic_view(self):
        device = self.opts.device
        # 1) 读取视频帧（numpy float32，[0,1]）
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

        # 2) VIPE 推理，得到每帧的 metric depth / intrinsics / pose(c2w)
        video_tensor = torch.from_numpy(frames_np)  # (T,H,W,3), float32 in [0,1]
        s = self.vipe.run(TensorVideoStream(video_tensor, fps=self.opts.fps, name="nvs_dynamic")).output_streams[0]
        depths_t = torch.stack([f.metric_depth for f in s]).to(device)         # (T,H,W)
        Ks_vipe = torch.stack([f.intrinsics for f in s]).to(device)            # (T, 4+D)
        Twcs_vipe = torch.stack([f.pose.matrix() for f in s]).to(device)       # (T,4,4), c2w (OpenCV)

        # 3) 计算第一帧前景中心深度作为半径，建立世界坐标系（与单图一致）
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

        # 4) 将 VIPE 的世界坐标映射到我们新的世界坐标：
        #    Twc_new_i = c2w_0 @ inv(Twc_vipe_0) @ Twc_vipe_i
        Twn_wv = c2w_0 @ torch.linalg.inv(Twcs_vipe[0])
        Twcs_new = Twn_wv.unsqueeze(0) @ Twcs_vipe  # (T,4,4)

        # 5) 每帧反投影到相机坐标 -> 映射到新世界坐标；同时准备颜色与深度掩码
        #    构造每帧的 3x3 内参矩阵
        fx, fy, cx, cy = Ks_vipe[:, 0], Ks_vipe[:, 1], Ks_vipe[:, 2], Ks_vipe[:, 3]
        K_3x3 = torch.zeros((T, 3, 3), device=device, dtype=torch.float32)
        K_3x3[:, 0, 0] = fx
        K_3x3[:, 1, 1] = fy
        K_3x3[:, 0, 2] = cx
        K_3x3[:, 1, 2] = cy
        K_3x3[:, 2, 2] = 1.0
        K_inv = torch.linalg.inv(K_3x3)

        # 像素网格 [u,v,1]
        uu, vv = torch.meshgrid(torch.arange(W, device=device, dtype=torch.float32),
                                 torch.arange(H, device=device, dtype=torch.float32), indexing="xy")
        uv1 = torch.stack([uu, vv, torch.ones_like(uu)], dim=-1).view(1, H * W, 3)  # (1,HW,3)
        uv1 = uv1.repeat(T, 1, 1)  # (T,HW,3)

        depths_hw = depths_t.view(T, H * W)  # (T,HW)

        points_world_list = []
        colors_list = []
        mask_list = []

        # 深度置信掩码（与单图一致的局部平滑性判断）
        depth_masks = self.reliable_depth_mask_range_batch(depths_t.unsqueeze(1), window_size=5, ratio_thresh=0.1)  # (T,1,H,W)

        frames_torch = torch.from_numpy(frames_np).to(device).to(torch.float32)  # (T,H,W,3) in [0,1]

        for i in range(T):
            Ki_inv = K_inv[i]                          # (3,3)
            rays = (Ki_inv @ uv1[i].T).T               # (HW,3)
            Xi_cam = rays * depths_hw[i][:, None]      # (HW,3)

            # 映射到新世界坐标：Xw = R * Xc + t
            R = Twcs_new[i][:3, :3]
            t = Twcs_new[i][:3, 3]
            Xi_world = (R @ Xi_cam.T).T + t[None]
            Xi_world = Xi_world.view(H, W, 3)

            points_world_list.append(Xi_world)
            # 颜色到 [-1,1]
            colors_list.append(frames_torch[i] * 2.0 - 1.0)  # (H,W,3)
            mask_list.append(depth_masks[i])                 # (1,H,W)

        points_world = torch.stack(points_world_list, dim=0)  # (T,H,W,3)
        colors = torch.stack(colors_list, dim=0)              # (T,H,W,3)
        masks = torch.stack(mask_list, dim=0).float()         # (T,1,H,W)

        # 6) 构造目标相机轨迹与内参（与单图一致，使用 build_cameras）
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
        # 移回设备以供渲染
        c2ws_target = c2ws_target.to(device)
        intrinsic_target = intrinsic_target.to(device)

        # 7) 渲染
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            control_imgs, render_masks = self.renderer.render(
                c2ws=c2ws_target,                # 目标相机 c2w
                Ks=intrinsic_target,            # 目标相机内参
                points_world_from_img1=points_world,
                colors_from_img1=colors,
                mask_img1=masks,
                normal_world_from_img1=None,
            )

        # 将渲染结果统一到扩散模型所需的分辨率（与 self.opts.height/width 一致）
        target_h, target_w = int(self.opts.height), int(self.opts.width)
        control_imgs = F.interpolate(control_imgs, size=[target_h, target_w], mode='bilinear', align_corners=False)
        render_masks = F.interpolate(render_masks.float(), size=[target_h, target_w], mode='nearest')

        # 8) 可视化与后续流程（对齐 nvs_single_view 的处理）
        # 将第 0 帧替换为原始图像
        control_imgs[0:1] = (frames_torch[0:1]).permute(0, 3, 1, 2) * 2.0 - 1.0
        render_masks[0:1] = 1.0

        frames_vis = frames_torch  # (T,H,W,3)
        save_video(
            frames_vis,
            os.path.join(self.opts.save_dir, 'input.mp4'),
            fps=self.opts.fps,
        )

        render_results = einops.rearrange(control_imgs, "f c h w -> f h w c", f=T)
        view_masks = einops.rearrange(render_masks, "f c h w -> f h w c", f=T)
        view_masks = 1.0 - view_masks

        save_video(
            (render_results + 1.0) / 2.0,
            os.path.join(self.opts.save_dir, 'render.mp4'),
            fps=self.opts.fps,
        )
        save_video(
            view_masks.repeat(1, 1, 1, 3),
            os.path.join(self.opts.save_dir, 'mask.mp4'),
            fps=self.opts.fps,
        )

        # 9) 送入扩散模型
        pil_mid = Image.fromarray((frames_np[T // 2] * 255).astype(np.uint8))
        prompt = self.get_caption(pil_mid)
        print(prompt)
        diffusion_results1 = self.run_diffusion(render_results, view_masks, prompt)
        diffusion_results = self.color_correct(diffusion_results1, (render_results + 1.0) / 2.0)
        save_video(diffusion_results1, os.path.join(self.opts.save_dir, 'diffusion_uncorrect.mp4'))
        save_video(diffusion_results, os.path.join(self.opts.save_dir, 'diffusion_correct.mp4'))

        # 拼接可视化
        tensor_left = frames_vis.to(device)
        tensor_right = diffusion_results.to(device)
        interval = torch.ones(T, self.opts.height, 30, 3, device=device)
        final_result = torch.cat((tensor_left, interval, tensor_right), dim=2)
        save_video(
            final_result,
            os.path.join(self.opts.save_dir, 'diffusion.mp4'),
            fps=self.opts.fps,
        )

    def run_dynamic_view_gradio(self,i2v_input_video, i2v_stride, i2v_elevation, i2v_center_scale, i2v_pose,i2v_steps, i2v_seed):
        # 若外部已设置 save_dir（app_dynamic.py 启动时已设置），则沿用，不再新建二级目录
        if not getattr(self.opts, "save_dir", None):
            prefix = datetime.now().strftime("%Y%m%d_%H%M")
            self.opts.save_dir = f'./output/gradio/{prefix}'
        os.makedirs(self.opts.save_dir, exist_ok=True)

        # fix seed
        self.opts.stride = i2v_stride
        self.opts.radius_scale = i2v_center_scale
        self.opts.image_dir = i2v_input_video
        self.opts.ddim_steps = i2v_steps
        self.opts.elevation = float(i2v_elevation)
        # 使用 build_cameras 的自定义轨迹方式，支持 swing1 标记或 5 段式参数
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
            # 提供默认值，避免缺项
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
                # 默认把第5个数当作 z 方向平移（与单图 UI 一致）
                # 如需作为半径变化 dr，可改为：self.opts.d_r = pose_vals[4]
                self.opts.z_offset = pose_vals[4]
        self.nvs_dynamic_view()
        # traj_dir = os.path.join(self.opts.save_dir, "viz_traj.mp4")
        gen_dir = os.path.join(self.opts.save_dir, "diffusion.mp4")
        return gen_dir #,traj_dir,


    def get_poses(self, opts, depths, num_frames):
        radius = (
            depths[0, 0, depths.shape[-2] // 2, depths.shape[-1] // 2].cpu()
            * opts.radius_scale
        )
        radius = min(radius, 5)
        cx = 512.0  # depths.shape[-1]//2 
        cy = 288.0  # depths.shape[-2]//2
        f = 500  # 500.
        K = (
            torch.tensor([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
            .repeat(num_frames, 1, 1)
            .to(opts.device)
        )
        # c2w_init = (
        #     torch.tensor(
        #         [
        #             [-1.0, 0.0, 0.0, 0.0],
        #             [0.0, 1.0, 0.0, 0.0],
        #             [0.0, 0.0, -1.0, 0.0],
        #             [0.0, 0.0, 0.0, 1.0],
        #         ]
        #     )
        #     .to(opts.device)
        #     .unsqueeze(0)
        # )

        # depths = -(depths - radius)

        # 初始化位置
        c2w_init = (
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            .to(opts.device)
            .unsqueeze(0)
        )

        dphi, dtheta, dx, dy, dr = opts.target_pose
        poses = generate_traj_specified(
            c2w_init, -dtheta, -dphi, dr * radius, dx, dy, num_frames, opts.device
        )

        poses[:, 2, 3] = poses[:, 2, 3] + radius
        pose_s = poses[opts.anchor_idx : opts.anchor_idx + 1].repeat(num_frames, 1, 1)
        pose_t = poses
        return pose_s, pose_t, K


    def setup_diffusion(self):

        # === Transformer ===
        transformer = WanVACETransformer3DModel.from_pretrained(
            self.opts.transformer_path,
            torch_dtype=self.opts.weight_dtype,
        )

        # self.transformer2 = WanVACETransformer3DModel.from_pretrained(
        #     self.opts.transformer_path_2,
        #     torch_dtype=self.opts.weight_dtype,
        # )

        # === VAE ===
        vae = AutoencoderKLWan.from_pretrained(self.opts.model_name, subfolder="vae", torch_dtype=torch.float32)


        # === Text encoder ===
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(self.opts.model_name, "tokenizer"),
        )
        text_encoder = UMT5EncoderModel.from_pretrained(
            os.path.join(self.opts.model_name, "text_encoder"),
            low_cpu_mem_usage=True,
            torch_dtype=self.opts.weight_dtype,
        )
        text_encoder = text_encoder.eval()


        # === Scheduler ===
        scheduler_config = {
            "num_train_timesteps": 1000,
            "shift": 5.0,
            "use_dynamic_shifting": False,
            "base_shift": 0.5,
            "max_shift": 1.15,
            "base_image_seq_len": 256,
            "max_image_seq_len": 4096
        }
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

        # === Pipeline ===
        self.pipeline = WanVACEPipeline(
            transformer=transformer,
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            scheduler=scheduler,
        )
        self.pipeline.to(self.opts.device)
        self.pipeline.load_lora_weights(self.opts.lora_path, adapter_name="causvid_lora")
        self.pipeline.set_adapters(["causvid_lora"], adapter_weights=[0.95])
        self.pipeline.fuse_lora()

 

    # def setup_dust3r(self):
    #     self.dust3r = load_model(self.opts.model_path, self.device)

    def setup_moge(self):
        self.depth_model = MoGeModel.from_pretrained(self.opts.moge_path).to(self.opts.device).eval()
        self.seg_net = TracerUniversalB7(device='cuda', batch_size=1,
                                    model_path=self.opts.segnet_path).eval()      
    
    def setup_vipe(self):
        # Initialize VIPE pipeline once; reuse for inference
        with hydra.initialize_config_dir(config_dir=str(get_config_path()), version_base=None):
            cfg = hydra.compose(
                "default",
                overrides=[
                    "pipeline=default",
                    "pipeline.post.depth_align_model=adaptive_unidepth-l_vda",
                    "pipeline.output.save_viz=false",
                    "pipeline.output.save_artifacts=false",
                    "pipeline.init.instance=null",
                    f"pipeline.output.path={self.opts.save_dir}/tmp_vipe",
                ],
            )
        self.vipe_cfg = cfg
        self.vipe = make_pipeline(cfg.pipeline)
        self.vipe.return_output_streams = True
        # Warper is still needed downstream
        self.funwarp = Warper(device=self.opts.device)

    # def load_initial_dir(self, image_dir):

    #     # 匹配 .png, .jpg, .jpeg 等（可扩展）
    #     image_files = glob.glob(os.path.join(image_dir, "*.[jJpP][pPnN][gG]")) + \
    #                 glob.glob(os.path.join(image_dir, "*.jpeg")) + \
    #                 glob.glob(os.path.join(image_dir, "*.JPEG"))  # 可选

    #     if len(image_files) < 2:
    #         raise ValueError("Input views should not less than 2.")
    #     image_pairs = []
    #     gt_pairs = []

    #     # image_files = sorted(image_files, key=lambda x: int(x.split('/')[-1].split('.')[0]))#[::2]
    #     for j in range(len(image_files)-1):
    #         images = load_images([image_files[j],image_files[j+1]], size=512,force_1024 = True)
    #         img_gts = [(images[0]['img_ori'].squeeze(0).permute(1,2,0)+1.)/2.,(images[1]['img_ori'].squeeze(0).permute(1,2,0)+1.)/2.]
    #         image_pairs.append(images)
    #         gt_pairs.append(img_gts)

    #     return image_pairs, gt_pairs

    def color_correct(self, diffusion_results: torch.Tensor, render_results: torch.Tensor) -> torch.Tensor:
        """
        颜色矫正：使用 render_results 第一帧作为参考，对 diffusion_results 所有帧进行颜色映射校正

        参数:
            diffusion_results: (f, h, w, c) torch.Tensor，值域 [0, 1]
            render_results:    (f, h, w, c) torch.Tensor，值域 [0, 1]

        返回:
            corrected_results: 同样 shape 和 dtype 的 torch.Tensor，颜色已校正
        """

        # 转 numpy，缩放到 [0, 255]
        gen_first = (diffusion_results[0].cpu().numpy() * 255).astype(np.float32)
        ref_first = (render_results[0].cpu().numpy() * 255).astype(np.float32)

        # reshape to N x 3
        src_flat = gen_first.reshape(-1, 3)
        tgt_flat = ref_first.reshape(-1, 3)

        # 构建仿射颜色映射 tgt = A * src + b
        X = np.hstack([src_flat, np.ones((src_flat.shape[0], 1))])
        Y = tgt_flat
        W, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        A = W[:3].T  # shape (3,3)
        b = W[3]     # shape (3,)

        # 对所有帧进行颜色校正
        corrected = []
        for i in range(diffusion_results.shape[0]):
            frame = (diffusion_results[i].cpu().numpy() * 255).astype(np.float32)
            h, w, c = frame.shape
            flat = frame.reshape(-1, 3)
            corrected_frame = flat @ A.T + b
            corrected_frame = np.clip(corrected_frame, 0, 255).reshape(h, w, 3).astype(np.uint8)
            corrected.append(corrected_frame)

        # 拼接为 numpy array，转换回 [0, 1] 的 torch tensor
        corrected_np = np.stack(corrected).astype(np.float32) / 255.0
        corrected_tensor = torch.from_numpy(corrected_np).to(diffusion_results.device).type(diffusion_results.dtype)
        return corrected_tensor

    def clean_points(self, warped_frame2, mask2):
        bs = mask2.shape[0]
        mask_new = []
        for i in range(bs):
            mask = mask2[i]
            mask = 1-mask
            mask[mask < 0.5] = 0
            mask[mask >= 0.5] = 1
            mask = mask.repeat(1,1,3)*255.
            mask = mask.cpu().numpy()
            kernel = np.ones((5,5), np.uint8)
            mask_erosion = cv2.dilate(np.array(mask), kernel, iterations = 1)
            mask_erosion = Image.fromarray(np.uint8(mask_erosion))
            mask_erosion_ = np.array(mask_erosion)/255.
            mask_erosion_[mask_erosion_ < 0.5] = 0
            mask_erosion_[mask_erosion_ >= 0.5] = 1
            mask_new.append(torch.from_numpy(mask_erosion_).to(self.device))

        mask_new = torch.stack(mask_new)
        warped_frame2 = warped_frame2*(1-mask_new)

        return warped_frame2, 1-mask_new[:,:,:,0:1]

    def get_caption(self,image):

        inputs = self.caption_processor(images=image, return_tensors="pt").to(self.opts.device, torch.float16)
        generated_ids = self.captioner.generate(**inputs)
        generated_text = self.caption_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip() 
        return generated_text + self.opts.refine_prompt    
    
    def reliable_depth_mask_range_batch(self, depth, window_size=5, ratio_thresh=0.05, eps=1e-6):
        assert window_size % 2 == 1, "Window size must be odd."
        if depth.dim() == 3:   # Input shape: (B, H, W)
            depth_unsq = depth.unsqueeze(1)
        elif depth.dim() == 4:  # Already has shape (B, 1, H, W)
            depth_unsq = depth
        else:
            raise ValueError("depth tensor must be of shape (B, H, W) or (B, 1, H, W)")
        
        local_max = torch.nn.functional.max_pool2d(depth_unsq, kernel_size=window_size, stride=1, padding=window_size // 2)
        local_min = -torch.nn.functional.max_pool2d(-depth_unsq, kernel_size=window_size, stride=1, padding=window_size // 2)
        local_mean = torch.nn.functional.avg_pool2d(depth_unsq, kernel_size=window_size, stride=1, padding=window_size // 2)
        ratio = (local_max - local_min) / (local_mean + eps)
        reliable_mask = (ratio < ratio_thresh) & (depth_unsq > 0)
        
        return reliable_mask
