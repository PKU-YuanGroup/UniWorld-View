import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
from torch.nn import functional as F
import numpy as np

from utils.torch_libs import ensure_torch_libs_loaded

ensure_torch_libs_loaded()

# --- PyTorch3D Imports ---
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    TexturesUV,
    hard_rgb_blend,
    BlendParams,
    PointsRenderer,
    PointsRasterizationSettings,
    PointsRasterizer,
    AlphaCompositor,
)
from pytorch3d.renderer.mesh.rasterizer import Fragments


def point_to_mesh_cuda(pts, rgbs, faces, min_angle_deg=2.5):
    h, w = rgbs.shape[:2]
    vertices = pts.reshape(-1, 3)
    # colors = rgbs.reshape(-1, 3)
    masks = torch.ones((h, w, 1), dtype=torch.uint8).to(rgbs.device) * 255
    rgbs = torch.cat([rgbs, masks], axis=-1)
    colors = rgbs.reshape(-1, 4)
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_normals = torch.cross(v1 - v0, v2 - v0)
    face_normals = F.normalize(face_normals, dim=1)
    def angle_between(v1, v2):
        cos_theta = torch.sum(v1 * v2, -1) / (
            torch.norm(v1, dim=1) * torch.norm(v2, dim=1) + 1e-12
        )
        return torch.arccos(torch.clip(cos_theta, -1.0, 1.0)) * 180 / np.pi

    a = angle_between(v1 - v0, v2 - v0)
    b = angle_between(v2 - v1, v0 - v1)
    c = angle_between(v0 - v2, v1 - v2)
    min_angles = torch.minimum(torch.minimum(a, b), c)

    # Filter faces based on minimum angle
    valid_faces = min_angles >= min_angle_deg
    z_range = vertices[:, 2].max() - vertices[:, 2].min()
    
    z01, z12, z20 = torch.abs((v0 - v1)[:, 2]), torch.abs((v1 - v2)[:, 2]), torch.abs((v2 - v0)[:, 2])
    y01, y12, y20 = torch.abs((v0 - v1)[:, 1]), torch.abs((v1 - v2)[:, 1]), torch.abs((v2 - v0)[:, 1])
    x01, x12, x20 = torch.abs((v0 - v1)[:, 0]), torch.abs((v1 - v2)[:, 0]), torch.abs((v2 - v0)[:, 0])
    z_max = torch.maximum(torch.maximum(z01, z12), z20)
    y_max = torch.maximum(torch.maximum(y01, y12), y20)
    x_max = torch.maximum(torch.maximum(x01, x12), x20)
    proj_max = torch.maximum(torch.maximum(x_max, y_max), z_max)
    valid_faces2 = (proj_max / z_range < 0.013)
    valid_faces_final = valid_faces & valid_faces2
    '''if args.use_dwmesh:
        invalid_faces = faces[~valid_faces]
        colors.index_put_((invalid_faces,), torch.zeros(4).to(colors.device))
    else:
        faces = faces[valid_faces]'''
    invalid_faces = faces[~valid_faces]
    colors.index_put_((invalid_faces,), torch.zeros(4).to(colors.device))

    return vertices, faces, colors


# --- 自定义点云渲染器 (来自您的代码) ---
class PointsZbufRenderer(PointsRenderer):
    """
    一个自定义的点云渲染器，除了渲染图像外，还返回z-buffer（深度图）。
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, point_clouds, **kwargs):
        fragments = self.rasterizer(point_clouds, **kwargs)

        r = self.rasterizer.raster_settings.radius
        dists2 = fragments.dists.permute(0, 3, 1, 2)
        weights = 1 - dists2 / (r * r)
        images = self.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            weights,
            point_clouds.features_packed().permute(1, 0),
            **kwargs,
        )

        images = images.permute(0, 2, 3, 1)

        # 返回渲染图像和z-buffer
        return images, fragments.zbuf


# --- 自定义网格着色器，用于输出深度 ---
class UnlitShaderWithDepth(nn.Module):
    def __init__(self, device="cpu", cameras=None, blend_params=None):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> Dict[str, torch.Tensor]:
        texels = meshes.sample_textures(fragments)
        colors = hard_rgb_blend(texels, fragments, self.blend_params)
        depth = fragments.zbuf[..., 0:1]
        
        return {
            "color": colors,
            "depth": depth
        }

# --- 主功能类 ---
class MeshWarperEx:
    def __init__(self, resolution: tuple = None, device: str = 'cpu'):
        self.resolution = resolution
        self.device = self.get_device(device)
        self.dtype = torch.float32

    def get_device(self, device: str):
        if torch.cuda.is_available() and 'cuda' in device:
            return torch.device(device)
        else:
            return torch.device('cpu')

    def create_mesh_from_depth(self, depth: torch.Tensor, intrinsic: torch.Tensor, frame: torch.Tensor, transformation1: torch.Tensor) -> Meshes:
        frame_normalized = (frame + 1.0) / 2.0

        b, _, h, w = depth.shape
        y, x = torch.meshgrid(torch.arange(h, device=self.device), torch.arange(w, device=self.device), indexing='ij')
        x, y = x.float(), y.float()
        
        fx, fy = intrinsic[:, 0, 0], intrinsic[:, 1, 1]
        cx, cy = intrinsic[:, 0, 2], intrinsic[:, 1, 2]
        
        z_cam = depth.squeeze(1)
        x_cam = (x - cx.view(b, 1, 1)) * z_cam / fx.view(b, 1, 1)
        y_cam = (y - cy.view(b, 1, 1)) * z_cam / fy.view(b, 1, 1)
        
        verts = torch.stack([x_cam, y_cam, z_cam], dim=-1).view(b, -1, 3)
        T_cam1_world = torch.inverse(transformation1)
        verts = torch.bmm(verts, T_cam1_world[:, :3, :3].transpose(1, 2)) + T_cam1_world[:, None, :3, 3]

        #point_to_mesh_cuda(verts, frame_normalized, faces)

        x_coords = torch.arange(w - 1, device=self.device)
        y_coords = torch.arange(h - 1, device=self.device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        xx, yy = xx.reshape(-1), yy.reshape(-1)
        idx = yy * w + xx
        
        # --- 修改开始 ---
        # 交换顶点的顺序以改变缠绕方向 (从顺时针改为逆时针，或者反之，确保法线指向相机)
        # 原代码: faces1 = torch.stack([idx, idx + 1, idx + w], dim=1)
        # 修改后: 交换后两个点
        faces1 = torch.stack([idx, idx + w, idx + 1], dim=1)

        # 原代码: faces2 = torch.stack([idx + 1, idx + w + 1, idx + w], dim=1)
        # 修改后: 交换后两个点
        faces2 = torch.stack([idx + 1, idx + w, idx + w + 1], dim=1)
        # --- 修改结束 ---
        faces = torch.cat([faces1, faces2], dim=0).unsqueeze(0).repeat(b, 1, 1)

        texture_map = frame_normalized.permute(0, 2, 3, 1)

        verts, faces, texture_map = point_to_mesh_cuda(verts[0], texture_map[0], faces[0])
        verts = verts.unsqueeze(0).repeat(b, 1, 1)
        faces = faces.unsqueeze(0).repeat(b, 1, 1)
        texture_map = texture_map[...,:3].reshape(1, h, w, 3).repeat(b, 1, 1, 1)

        verts_uvs = torch.stack([
            x.view(-1) / (w - 1), 
            1.0 - y.view(-1) / (h - 1)
        ], dim=-1).unsqueeze(0).repeat(b, 1, 1)
        
        textures = TexturesUV(maps=texture_map, faces_uvs=faces.clone(), verts_uvs=verts_uvs)

        return Meshes(verts=verts, faces=faces, textures=textures)

    def forward_warp(
        self,
        frame1: torch.Tensor,
        depth1: torch.Tensor,
        transformation1: torch.Tensor,
        transformation2: torch.Tensor,
        intrinsic1: torch.Tensor,
        intrinsic2: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.resolution is not None:
            assert frame1.shape[2:4] == self.resolution
        
        b, _, h, w = frame1.shape
        if intrinsic2 is None:
            intrinsic2 = intrinsic1.clone()

        mesh_world = self.create_mesh_from_depth(depth1, intrinsic1, frame1, transformation1)

        # transformation1 是 world-to-cam1 的变换矩阵，所以 inverse(transformation1) 是 cam1-to-world
        #mesh_world = Meshes(verts=verts_world, faces=mesh.faces_padded(), textures=mesh.textures)

        # transformation2 是 world-to-cam2 的变换矩阵
        R_cv, T_cv = transformation2[:, :3, :3], transformation2[:, :3, 3]

        # 转换到 PyTorch3D 的相机坐标系 (NDC, z朝后)
        R_p3d = R_cv.clone().transpose(1, 2)
        T_p3d = T_cv.clone()
        R_p3d[:, :, 0] *= -1
        R_p3d[:, :, 1] *= -1
        T_p3d[:, 0] *= -1
        T_p3d[:, 1] *= -1

        # prepare camera intrinsics for PyTorch3D
        fx = intrinsic2[:, 0, 0].to(device=self.device, dtype=self.dtype)
        fy = intrinsic2[:, 1, 1].to(device=self.device, dtype=self.dtype)
        cx = intrinsic2[:, 0, 2].to(device=self.device, dtype=self.dtype)
        cy = intrinsic2[:, 1, 2].to(device=self.device, dtype=self.dtype)
        
        cameras = PerspectiveCameras(
            focal_length=torch.stack([fx, fy], dim=1),
            principal_point=torch.stack([cx, cy], dim=1),
            R=R_p3d,
            T=T_p3d,
            image_size=((h, w),),
            in_ndc=False,
            device=self.device,
        )

        raster_settings = RasterizationSettings(
            image_size=(h, w), blur_radius=0.0, faces_per_pixel=1, cull_backfaces=True
        )
        blend_params = BlendParams(background_color=(0.0, 0.0, 0.0))

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=UnlitShaderWithDepth(device=self.device, cameras=cameras, blend_params=blend_params)
        )
        
        rendered_output_dict = renderer(mesh_world)
        rendered_color = rendered_output_dict["color"]
        rendered_depth = rendered_output_dict["depth"]

        rendered_output_rescaled = rendered_color * 2.0 - 1.0
        rendered_output_permuted = rendered_output_rescaled.permute(0, 3, 1, 2)
        
        warped_frame2 = rendered_output_permuted[:, :3, :, :]
        mask2 = (rendered_output_permuted[:, 3:, :, :] > -1.0).float() 
        warped_depth2 = rendered_depth.permute(0, 3, 1, 2)
        
        return warped_frame2, mask2, warped_depth2

    # --- 修改后的点云渲染方法 ---
    def render_point_cloud(
        self,
        frame1: torch.Tensor,
        depth1: torch.Tensor,
        transformation1: torch.Tensor,
        transformation2: torch.Tensor,
        intrinsic1: torch.Tensor,
        intrinsic2: Optional[torch.Tensor] = None,
        # --- 点云专属参数作为可选关键字参数 ---
        point_radius: float = 0.01,
        points_per_pixel: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        利用网格顶点和纹理，从一个新的视角渲染点云。
        该方法的输入参数与 forward_warp 完全一致，以保证接口统一。
        """
        b, _, h, w = frame1.shape
        if intrinsic2 is None:
            intrinsic2 = intrinsic1.clone()

        # 1. 从源视图(1)的深度图创建网格，以获取顶点和颜色
        mesh_cam1 = self.create_mesh_from_depth(depth1, intrinsic1, frame1)
        verts_cam1 = mesh_cam1.verts_padded()
        # 将纹理图 (B, H, W, 3) 展平为每个顶点的颜色 (B, H*W, 3)
        colors = mesh_cam1.textures.maps_padded().view(b, -1, 3)
        colors = (frame1.permute(0, 2, 3, 1).view(b, -1, 3) + 1.0) / 2.0

        # 2. 将顶点从相机坐标系1转换到世界坐标系
        # transformation1 是 world-to-cam1, 所以其逆是 cam1-to-world
        T_cam1_world = torch.inverse(transformation1)
        verts_world = torch.bmm(verts_cam1, T_cam1_world[:, :3, :3].transpose(1, 2)) + T_cam1_world[:, None, :3, 3]

        # 3. 设置目标视角(2)的PyTorch3D相机 (与 forward_warp 完全相同)
        R_cv, T_cv = transformation2[:, :3, :3], transformation2[:, :3, 3]

        R_p3d = R_cv.clone().transpose(1, 2)
        T_p3d = T_cv.clone()
        R_p3d[:, :, 0] *= -1
        R_p3d[:, :, 1] *= -1
        T_p3d[:, 0] *= -1
        T_p3d[:, 1] *= -1
        
        # prepare camera intrinsics for PyTorch3D
        fx = intrinsic2[:, 0, 0].to(device=self.device, dtype=self.dtype)
        fy = intrinsic2[:, 1, 1].to(device=self.device, dtype=self.dtype)
        cx = intrinsic2[:, 0, 2].to(device=self.device, dtype=self.dtype)
        cy = intrinsic2[:, 1, 2].to(device=self.device, dtype=self.dtype)

        cameras = PerspectiveCameras(
            focal_length=torch.stack([fx, fy], dim=1),
            principal_point=torch.stack([cx, cy], dim=1),
            R=R_p3d,
            T=T_p3d,
            image_size=((h, w),),
            in_ndc=False,
            device=self.device,
        )

        # 4. 创建点云对象
        point_cloud = Pointclouds(points=[verts_world[i] for i in range(b)], features=[colors[i] for i in range(b)])

        # 5. 设置点云渲染器
        raster_settings = PointsRasterizationSettings(
            image_size=(h, w), 
            radius=point_radius,
            points_per_pixel=points_per_pixel,
        )

        renderer = PointsZbufRenderer(
            rasterizer=PointsRasterizer(cameras=cameras, raster_settings=raster_settings),
            compositor=AlphaCompositor(background_color=(0.0, 0.0, 0.0))
        )

        # 6. 执行渲染
        rendered_images, rendered_zbuf = renderer(point_cloud)

        # 7. 处理并返回输出
        rendered_images_rescaled = rendered_images * 2.0 - 1.0
        rendered_images_permuted = rendered_images_rescaled.permute(0, 3, 1, 2)[:, :3, :, :]
        
        rendered_depth = rendered_zbuf[..., 0:1].permute(0, 3, 1, 2)
        mask = (rendered_depth > 0).float()
        
        return rendered_images_permuted, mask, rendered_depth
