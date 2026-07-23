import torch
import numpy as np
import cv2
import PIL
from typing import Optional, Tuple
import torch.nn.functional as F

class ImageSplattingRenderer:
    def __init__(self, resolution: tuple = None, device: str = 'cuda:0'):
        self.resolution = resolution
        self.device =device
        self.dtype = torch.float32

    @torch.no_grad()
    def render(
        self,
        c2ws: torch.Tensor,                   # (B,4,4)  目标相机 c2w（camera->world）
        Ks: torch.Tensor,                     # (B,3,3)  目标相机内参（像素单位）
        points_world_from_img1: torch.Tensor, # (B,3,H,W)  由 image1 反投影并转到世界坐标系的点云
        colors_from_img1: torch.Tensor,       # (B,3,H,W)  image1 上的颜色
        normal_world_from_img1: Optional[torch.Tensor] = None,  # (B,3,H,W)  image1 上的法线
        mask_img1: Optional[torch.Tensor] = None,  # (B,1,H,W)  image1 上的掩码
        vis_threshold: float = -0.1,
        clean = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        把世界坐标点云投影到一组目标相机，返回:
          trans_coordinates: (N,H,W,2)  每个相机下的像素坐标 (u,v)
          trans_depth1:      (N,H,W)    相机坐标深度 Z
          flow12:            (N,2,H,W)  从 image1 到各目标相机的光流 (du,dv)
        说明:
          flow12[n,y,x] = trans_coordinates[n,y,x] - (x,y)
          注意：这里源网格使用整数像素角点 (x,y)，不加 0.5 的像素中心偏移。
        """
        device = self.device
        dtype  = self.dtype
        points_world_from_img1 = points_world_from_img1.to(device, dtype)
        colors_from_img1 = colors_from_img1.to(device, dtype)
        c2ws = c2ws.to(device, dtype)
        Ks = Ks.to(device, dtype)

        N = c2ws.shape[0]
        H, W = points_world_from_img1.shape[1:3]

        # if self.resolution is not None:
        #     assert (H, W) == self.resolution, "points_world_from_img1 分辨率与类初始化不一致"

        if mask_img1 is None:
            mask_img1 = torch.ones(N, 1, H, W).to(c2ws)

        if normal_world_from_img1 is not None:
            normal_world_from_img1 = normal_world_from_img1.to(device, dtype)
            points_3d = points_world_from_img1.reshape(N, H * W, 3)  # (N, H*W, 3)
            normal = normal_world_from_img1.reshape(N, H * W, 3)
            camera_centers = c2ws[:, :3, 3]
            # View direction: from camera to 3D point
            view_dirs = F.normalize(points_3d - camera_centers[:, None, :], dim=-1)  # [F, N, 3]
            # Flip view_dirs so that it's from point to camera
            view_dirs = -view_dirs  
            # Calculate visibility mask based on the angle between the surface normal and view direction
            cos_map = torch.sum(normal * view_dirs, dim=-1)
            threshold = 0. #vis_threshold
            vis_mask = (cos_map > threshold)
            mask_img1  = vis_mask.view(N, 1, H, W) * mask_img1 # Apply visibility mask to the original mask

        colors_from_img1 = colors_from_img1.permute(0, 3, 1, 2)  # (N,3,H,W)

        # ---- 世界点齐次，并广播到每个相机 ----
        ones = torch.ones(N, H, W, 1, device=device, dtype=dtype)
        Pw = torch.cat([points_world_from_img1, ones], dim=-1)      # (H,W,4)
        Pw = Pw.unsqueeze(-1)                                       # (H,W,4,1), column vector

        # ---- 世界->相机：w2c = inv(c2w) ----
        w2c = torch.linalg.inv(c2ws).to(dtype)                      # (N,4,4)
        w2c = w2c[:, None, None, :, :]                              # (N,1,1,4,4)

        Pc_h = w2c @ Pw                                             # (N,H,W,4,1)
        Pc   = Pc_h[..., :3, :]                                     # (N,H,W,3,1), points in camera coord
        Z    = Pc[..., 2, 0]                                        # (N,H,W), depth in camera coord

        # ---- 像素投影：uv_h = K @ Pc，然后做透视除法 ----
        Kb = Ks[:, None, None, :, :]                                # (N,1,1,3,3)
        uv_h = Kb @ Pc                                              # (N,H,W,3,1) -> (u',v',Z)
        # Z_safe = Z.clamp(min=eps)                                   # (N,H,W)
        trans_coordinates = (uv_h[..., :2, 0] / Z.unsqueeze(-1))  # (N,H,W,2)

        # ---- 源图像网格（不加 0.5）----
        src_grid = self.create_grid(1, H, W).permute(0, 2, 3, 1).to(trans_coordinates)  # (1, H, W, 2)      

        # ---- 光流：目标像素坐标 - 源像素坐标 ----
        flow = trans_coordinates - src_grid                         # (N,H,W,2)
        flow12 = flow.permute(0, 3, 1, 2)                           # (N,2,H,W)

        trans_depth1 = Z                                            # (N,H,W)
        mask_img1 = mask_img1 * (trans_depth1[:,None,:,:] > 0)

        render_results, render_masks = self.bilinear_splatting(
            colors_from_img1, mask_img1, trans_depth1, flow12, None, is_image=True
        )

        '''render_results, render_masks = self.nearest_splatting(
            colors_from_img1, mask_img1, trans_depth1, flow12, None, is_image=True
        )'''

        depth_img = trans_depth1.unsqueeze(1)          # (N,H,W) -> (N,1,H,W)
        # 2. 用与颜色完全相同的坐标流 flow12 做 splatting
        depth_results, depth_mask = self.bilinear_splatting(
            depth_img,              # 1 通道“图像”
            mask_img1,              # 沿用同一个 mask
            trans_depth1,           # 实际深度值（函数里用于遮挡判断，可复用）
            flow12,                 # 同一套光流
            None,
            is_image=False          # 深度是标量，不是彩色，用 False 可避免归一化
        )

        # 3. 如果想把无效区域填 0（或 NaN）
        #warped_depth = warped_depth * depth_mask

        #if clean:
        #    render_results, render_masks = self.clean_points(render_results, render_masks)
        #    #depth_results, depth_mask = self.clean_points(depth_results, depth_mask)
        
        return render_results, render_masks, depth_results, depth_mask


    def bilinear_splatting(
        self,
        frame1: torch.Tensor,
        mask1: Optional[torch.Tensor],
        depth1: torch.Tensor,
        flow12: torch.Tensor,
        flow12_mask: Optional[torch.Tensor],
        is_image: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Bilinear splatting
        :param frame1: (b,c,h,w)
        :param mask1: (b,1,h,w): 1 for known, 0 for unknown. Optional
        :param depth1: (b,1,h,w)
        :param flow12: (b,2,h,w)
        :param flow12_mask: (b,1,h,w): 1 for valid flow, 0 for invalid flow. Optional
        :param is_image: if true, output will be clipped to (-1,1) range
        :return: warped_frame2: (b,c,h,w)
                 mask2: (b,1,h,w): 1 for known and 0 for unknown
        """
        # if self.resolution is not None:
        #     assert frame1.shape[2:4] == self.resolution
        b, c, h, w = frame1.shape
        if mask1 is None:
            mask1 = torch.ones(size=(b, 1, h, w)).to(frame1)
        if flow12_mask is None:
            flow12_mask = torch.ones(size=(b, 1, h, w)).to(flow12)
        grid = self.create_grid(b, h, w).to(frame1)
        trans_pos = flow12 + grid

        trans_pos_offset = trans_pos + 1
        trans_pos_floor = torch.floor(trans_pos_offset).long()
        trans_pos_ceil = torch.ceil(trans_pos_offset).long()
        trans_pos_offset = torch.stack(
            [
                torch.clamp(trans_pos_offset[:, 0], min=0, max=w + 1),
                torch.clamp(trans_pos_offset[:, 1], min=0, max=h + 1),
            ],
            dim=1,
        )
        trans_pos_floor = torch.stack(
            [
                torch.clamp(trans_pos_floor[:, 0], min=0, max=w + 1),
                torch.clamp(trans_pos_floor[:, 1], min=0, max=h + 1),
            ],
            dim=1,
        )
        trans_pos_ceil = torch.stack(
            [
                torch.clamp(trans_pos_ceil[:, 0], min=0, max=w + 1),
                torch.clamp(trans_pos_ceil[:, 1], min=0, max=h + 1),
            ],
            dim=1,
        )

        prox_weight_nw = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * (
            1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1])
        )
        prox_weight_sw = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * (
            1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1])
        )
        prox_weight_ne = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * (
            1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1])
        )
        prox_weight_se = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * (
            1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1])
        )

        sat_depth1 = torch.clamp(depth1, min=0, max=1000)
        log_depth1 = torch.log(1 + sat_depth1)
        depth_weights = torch.exp(log_depth1 / log_depth1.max() * 50)

        weight_nw = torch.moveaxis(
            prox_weight_nw * mask1 * flow12_mask / depth_weights.unsqueeze(1),
            [0, 1, 2, 3],
            [0, 3, 1, 2],
        )
        weight_sw = torch.moveaxis(
            prox_weight_sw * mask1 * flow12_mask / depth_weights.unsqueeze(1),
            [0, 1, 2, 3],
            [0, 3, 1, 2],
        )
        weight_ne = torch.moveaxis(
            prox_weight_ne * mask1 * flow12_mask / depth_weights.unsqueeze(1),
            [0, 1, 2, 3],
            [0, 3, 1, 2],
        )
        weight_se = torch.moveaxis(
            prox_weight_se * mask1 * flow12_mask / depth_weights.unsqueeze(1),
            [0, 1, 2, 3],
            [0, 3, 1, 2],
        )

        warped_frame = torch.zeros(size=(b, h + 2, w + 2, c), dtype=torch.float32).to(
            frame1
        )
        warped_weights = torch.zeros(size=(b, h + 2, w + 2, 1), dtype=torch.float32).to(
            frame1
        )

        frame1_cl = torch.moveaxis(frame1, [0, 1, 2, 3], [0, 3, 1, 2])
        batch_indices = torch.arange(b)[:, None, None].to(frame1.device)
        warped_frame.index_put_(
            (batch_indices, trans_pos_floor[:, 1], trans_pos_floor[:, 0]),
            frame1_cl * weight_nw,
            accumulate=True,
        )
        warped_frame.index_put_(
            (batch_indices, trans_pos_ceil[:, 1], trans_pos_floor[:, 0]),
            frame1_cl * weight_sw,
            accumulate=True,
        )
        warped_frame.index_put_(
            (batch_indices, trans_pos_floor[:, 1], trans_pos_ceil[:, 0]),
            frame1_cl * weight_ne,
            accumulate=True,
        )
        warped_frame.index_put_(
            (batch_indices, trans_pos_ceil[:, 1], trans_pos_ceil[:, 0]),
            frame1_cl * weight_se,
            accumulate=True,
        )

        warped_weights.index_put_(
            (batch_indices, trans_pos_floor[:, 1], trans_pos_floor[:, 0]),
            weight_nw,
            accumulate=True,
        )
        warped_weights.index_put_(
            (batch_indices, trans_pos_ceil[:, 1], trans_pos_floor[:, 0]),
            weight_sw,
            accumulate=True,
        )
        warped_weights.index_put_(
            (batch_indices, trans_pos_floor[:, 1], trans_pos_ceil[:, 0]),
            weight_ne,
            accumulate=True,
        )
        warped_weights.index_put_(
            (batch_indices, trans_pos_ceil[:, 1], trans_pos_ceil[:, 0]),
            weight_se,
            accumulate=True,
        )

        warped_frame_cf = torch.moveaxis(warped_frame, [0, 1, 2, 3], [0, 2, 3, 1])
        warped_weights_cf = torch.moveaxis(warped_weights, [0, 1, 2, 3], [0, 2, 3, 1])
        cropped_warped_frame = warped_frame_cf[:, :, 1:-1, 1:-1]
        cropped_weights = warped_weights_cf[:, :, 1:-1, 1:-1]

        mask = cropped_weights > 0
        zero_value = -1 if is_image else 0
        zero_tensor = torch.tensor(zero_value, dtype=frame1.dtype, device=frame1.device)
        warped_frame2 = torch.where(
            mask, cropped_warped_frame / cropped_weights, zero_tensor
        )
        mask2 = mask.to(frame1)

        if is_image:
            assert warped_frame2.min() >= -1.1  # Allow for rounding errors
            assert warped_frame2.max() <= 1.1
            warped_frame2 = torch.clamp(warped_frame2, min=-1, max=1)
        return warped_frame2, mask2

    def nearest_splatting(
        self,
        frame1: torch.Tensor,
        mask1: Optional[torch.Tensor],
        depth1: torch.Tensor,
        flow12: torch.Tensor,
        flow12_mask: Optional[torch.Tensor],
        is_image: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = frame1.shape
        device = frame1.device

        if mask1 is None:
            mask1 = torch.ones(b, 1, h, w, device=device)
        if flow12_mask is None:
            flow12_mask = torch.ones(b, 1, h, w, device=device)

        grid = self.create_grid(b, h, w).to(device)
        trans_pos = flow12 + grid  # (b,2,h,w)

        # round to nearest pixel
        trans_pos_round = torch.round(trans_pos).long()  # (b,2,h,w)

        # clamp to image bounds
        x = trans_pos_round[:, 0].clamp(0, w - 1)  # (b,h,w)
        y = trans_pos_round[:, 1].clamp(0, h - 1)  # (b,h,w)

        # prepare output buffers
        warped_frame = torch.zeros(b, h, w, c, device=device, dtype=torch.float32)
        warped_weights = torch.zeros(b, h, w, 1, device=device, dtype=torch.float32)

        # flatten spatial dims for indexing
        flat_index = y * w + x  # (b,h,w)

        # frame1 in (b,h,w,c) order
        frame1_p = frame1.permute(0, 2, 3, 1)

        # depth-based weight (optional, keep if you want)
        sat_depth1 = torch.clamp(depth1, 0, 1000)
        log_depth1 = torch.log1p(sat_depth1)
        depth_weights = torch.exp(log_depth1 / log_depth1.max() * 50)  # (b,1,h,w)
        contrib_weight = (mask1 * flow12_mask / depth_weights).permute(0, 2, 3, 1)  # (b,h,w,1)

        # accumulate
        for i in range(b):
            warped_frame[i].view(-1, c).index_add_(0, flat_index[i].view(-1), (frame1_p[i] * contrib_weight[i]).view(-1, c))
            warped_weights[i].view(-1).index_add_(0, flat_index[i].view(-1), contrib_weight[i].view(-1))

        # normalize
        mask2 = warped_weights.squeeze(-1) > 0
        zero_val = -1 if is_image else 0
        warped_frame2 = torch.where(mask2.unsqueeze(-1), warped_frame / warped_weights.clamp_min(1e-8), zero_val)
        warped_frame2 = warped_frame2.permute(0, 3, 1, 2)  # (b,c,h,w)
        mask2 = mask2.unsqueeze(1)  # (b,1,h,w)

        if is_image:
            warped_frame2 = torch.clamp(warped_frame2, -1, 1)

        return warped_frame2, mask2

    def create_grid(self, b, h, w):
        # 生成坐标网格
        y, x = torch.meshgrid(
            torch.arange(h),
            torch.arange(w),
            indexing='ij'   # 保证第一个是行(y)，第二个是列(x)
        )  # y.shape, x.shape = (h, w)

        grid = torch.stack([x, y], dim=0)  # (2, h, w)
        batch_grid = grid.unsqueeze(0).repeat(b, 1, 1, 1)  # (b, 2, h, w)
        return batch_grid

    def clean_points(self, warped_frame2, mask2):
        warped_frame2 = (warped_frame2 + 1.0) / 2.0
        mask = 1 - mask2
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = mask.squeeze(0).repeat(3, 1, 1).permute(1, 2, 0) * 255.0
        mask = mask.cpu().numpy()
        kernel = np.ones((3, 3), np.uint8)
        mask_erosion = cv2.dilate(np.array(mask), kernel, iterations=1)
        mask_erosion = PIL.Image.fromarray(np.uint8(mask_erosion))
        mask_erosion_ = np.array(mask_erosion) / 255.0
        mask_erosion_[mask_erosion_ < 0.5] = 0
        mask_erosion_[mask_erosion_ >= 0.5] = 1
        mask_new = (
            torch.from_numpy(mask_erosion_)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )
        warped_frame2 = warped_frame2 * (1 - mask_new)
        return warped_frame2 * 2.0 - 1.0, 1 - mask_new[:, 0:1, :, :]

if __name__ == '__main__':

    # Example usage
    H = 480  # Example height
    W = 832
    renderer = ImageSplattingRenderer(resolution=(H, W), device='cuda:0')
    points = torch.randn(81,H, W, 3).to('cuda:0')
    colors = (torch.rand(81, H, W,3) * 2 - 1).to('cuda:0')
    c2ws = torch.eye(4).unsqueeze(0).repeat(81, 1, 1).to('cuda:0')  # Example camera pose
    Ks = torch.eye(3).unsqueeze(0).repeat(81, 1, 1).to('cuda:0')  # Example camera intrinsics

    rendered, masks = renderer.render(
        c2ws=c2ws,
        Ks=Ks,
        points_world_from_img1=points,
        colors_from_img1=colors,
        mask_img1=None,  # Optional mask
        normal_world_from_img1=None,  # Optional normals
    )
    print(rendered.shape, masks.shape)  # Should print the shapes of the rendered image and masks
