import contextlib
import os
import sys

import numpy as np
import einops
import kornia
import torch

from utils.torch_libs import ensure_torch_libs_loaded

ensure_torch_libs_loaded()

from pytorch3d.renderer import (
    PointsRenderer,
    PointsRasterizationSettings,
    PointsRasterizer,
    AlphaCompositor,
    PerspectiveCameras,
)
from pytorch3d.structures import Pointclouds

from utils.utils import points_padding


def get_boundaries_mask(disparity, sobel_threshold=0.3):
    def sobel_filter(disp, mode="sobel", beta=10.0):
        sobel_grad = kornia.filters.spatial_gradient(disp, mode=mode, normalized=False)
        sobel_mag = torch.sqrt(sobel_grad[:, :, 0, Ellipsis] ** 2 + sobel_grad[:, :, 1, Ellipsis] ** 2)
        alpha = torch.exp(-1.0 * beta * sobel_mag).detach()

        return alpha

    sobel_beta = 10.0
    normalized_disparity = (disparity - disparity.min()) / (disparity.max() - disparity.min() + 1e-6)
    return sobel_filter(normalized_disparity, "sobel", beta=sobel_beta) < sobel_threshold


class PointsZbufRenderer(PointsRenderer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, point_clouds, **kwargs):
        fragments = self.rasterizer(point_clouds, **kwargs)

        # Construct weights based on the distance of a point to the true point.
        # However, this could be done differently: e.g. predicted as opposed
        # to a function of the weights.
        r = self.rasterizer.raster_settings.radius

        dists2 = fragments.dists.permute(0, 3, 1, 2)
        weights = 1 - dists2 / (r * r)
        images = self.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            weights,
            point_clouds.features_packed().permute(1, 0),
            **kwargs,
        )

        # permute so image comes at the end
        images = images.permute(0, 2, 3, 1)

        return images, fragments.zbuf


@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stdout_fd = os.dup(sys.stdout.fileno())
        old_stderr_fd = os.dup(sys.stderr.fileno())
        try:
            os.dup2(devnull.fileno(), sys.stdout.fileno())
            os.dup2(devnull.fileno(), sys.stderr.fileno())
            yield
        finally:
            os.dup2(old_stdout_fd, sys.stdout.fileno())
            os.dup2(old_stderr_fd, sys.stderr.fileno())
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)


def point_rendering(K, w2cs, depth, image, raster_settings, device,
                    background_color=[0, 0, 0], sobel_threshold=0.35, contract=8.0,
                    sam_mask=None):
    """
    only support batchsize=1
    :param K: [F,3,3]
    :param w2cs: [F,4,4] opencv
    :param depth: [1,1,H,W]
    :param images: [1,3,H,W]
    :param background_color: [-1,-1,-1]~[1,1,1]
    :param raster_settings:
    :param sam_mask: [1,1,H,W] 0 or 1
    :return: render_rgbs, render_masks
    """
    nframe = w2cs.shape[0]
    _, _, h, w = image.shape

    # depth contract
    depth = depth.to(device)
    K = K.to(device)
    w2cs = w2cs.to(device)
    image = image.to(device)
    c2ws = w2cs.inverse()

    if depth.max() == 0:
        render_rgbs = torch.zeros((nframe, 3, h, w), device=device, dtype=torch.float32)
        render_masks = torch.ones((nframe, 1, h, w), device=device, dtype=torch.float32)
    else:
        mid_depth = torch.median(depth.reshape(-1), dim=0)[0] * contract
        depth[depth > mid_depth] = ((2 * mid_depth) - (mid_depth ** 2 / (depth[depth > mid_depth] + 1e-6)))

        point_depth = einops.rearrange(depth[0], "c h w -> (h w) c")
        disp = 1 / (depth + 1e-7)
        boundary_mask = get_boundaries_mask(disp, sobel_threshold=sobel_threshold)

        x = torch.arange(w).float() + 0.5
        y = torch.arange(h).float() + 0.5
        points = torch.stack(torch.meshgrid(x, y, indexing='ij'), -1).to(device)
        points = einops.rearrange(points, "w h c -> (h w) c")
        # GPU求逆有错
        points_3d = (c2ws[0] @ points_padding((K[0].cpu().inverse().to(device) @ points_padding(points).T).T * point_depth).T).T[:, :3]

        colors = einops.rearrange(image[0], "c h w -> (h w) c")

        boundary_mask = boundary_mask.reshape(-1)
        if sam_mask is not None:
            sam_mask = sam_mask.reshape(-1)
            boundary_mask[sam_mask == True] = True

        points_3d = points_3d[boundary_mask == False]

        if points_3d.shape[0] <= 8:
            render_rgbs = torch.zeros((nframe, 3, h, w), device=device, dtype=torch.float32)
            render_masks = torch.ones((nframe, 1, h, w), device=device, dtype=torch.float32)
            render_rgbs[0:1] = image
            render_masks[0:1] = 0
            return render_rgbs, render_masks

        colors = colors[boundary_mask == False]

        point_cloud = Pointclouds(points=[points_3d.to(device)], features=[colors.to(device)]).extend(nframe)

        # convert opencv to opengl coordinate
        c2ws[:, :, 0] = - c2ws[:, :, 0]
        c2ws[:, :, 1] = - c2ws[:, :, 1]
        w2cs = c2ws.inverse()

        focal_length = torch.stack([K[:, 0, 0], K[:, 1, 1]], dim=1)
        principal_point = torch.stack([K[:, 0, 2], K[:, 1, 2]], dim=1)
        image_shapes = torch.tensor([[h, w]]).repeat(nframe, 1)
        cameras = PerspectiveCameras(focal_length=focal_length, principal_point=principal_point,
                                     R=c2ws[:, :3, :3], T=w2cs[:, :3, -1], in_ndc=False,
                                     image_size=image_shapes, device=device)

        renderer = PointsZbufRenderer(
            rasterizer=PointsRasterizer(cameras=cameras, raster_settings=raster_settings),
            compositor=AlphaCompositor(background_color=background_color)
        )



        try:
            with suppress_stdout_stderr():
                render_rgbs, zbuf = renderer(point_cloud)  # rgb:[f,h,w,3]
        except Exception as e:
            print(f"Error: {e}")
            print("Error rendering, save pointcloud and other data...")
            torch.save(points_3d, "point_3d_debug.pt")
            torch.save(colors, "colors_debug.pt")
            torch.save(boundary_mask, "boundary_mask_debug.pt")

        render_masks = (zbuf[..., 0:1] == -1).float()  # [f,h,w,1]
        render_rgbs = einops.rearrange(render_rgbs, "f h w c -> f c h w")  # [f,3,h,w]
        render_masks = einops.rearrange(render_masks, "f h w c -> f c h w")  # [f,1,h,w]

    # replace the first frame
    render_rgbs[0:1] = image
    render_masks[0:1] = 0

    return render_rgbs, render_masks


def render_pcd(pts3d, imgs, masks, views, renderer, device, return_mask=False, return_depth=False):
    imgs_np = imgs.detach().cpu().numpy()
    pts3d_np = pts3d.detach().cpu().numpy()

    if masks is None:
        pts = torch.from_numpy(np.concatenate([p for p in pts3d_np])).view(-1, 3).to(device)
        col = torch.from_numpy(np.concatenate([p for p in imgs_np])).view(-1, 3).to(device)
    else:
        masks_np = masks.detach().cpu().numpy().astype(bool)
        pts = torch.from_numpy(np.concatenate([p[m] for p, m in zip(pts3d_np, masks_np)])).view(-1, 3).to(device)
        col = torch.from_numpy(np.concatenate([p[m] for p, m in zip(imgs_np, masks_np)])).view(-1, 3).to(device)

    point_cloud = Pointclouds(points=[pts], features=[col]).extend(views)
    images = renderer(point_cloud)[0]

    if return_depth:
        fragments = renderer.rasterizer(point_cloud)
        depth_maps = fragments.zbuf[..., 0]
        depth_maps = torch.where(
            depth_maps >= 0,
            depth_maps,
            torch.tensor(float("inf"), device=device),
        )
    else:
        fragments = None
        depth_maps = None

    if return_mask:
        if fragments is None:
            fragments = renderer.rasterizer(point_cloud)
        view_masks = (fragments.zbuf[..., 0] > 0).float()
    else:
        view_masks = None

    return images, view_masks, depth_maps


def run_render(pcd, imgs, masks, H, W, c2ws, K, num_views, return_mask=False, return_depth=False, device="cuda:0"):
    return_depth = True

    c2ws = c2ws.clone()
    c2ws[:, :, 0] = -c2ws[:, :, 0]
    c2ws[:, :, 1] = -c2ws[:, :, 1]
    w2cs = c2ws.inverse()

    background_color = [0, 0, 0]
    focal_length = torch.stack([K[:, 0, 0], K[:, 1, 1]], dim=1)
    principal_point = torch.stack([K[:, 0, 2], K[:, 1, 2]], dim=1)
    image_shapes = torch.tensor([[H, W]]).repeat(num_views, 1)
    cameras = PerspectiveCameras(
        focal_length=focal_length,
        principal_point=principal_point,
        R=c2ws[:, :3, :3],
        T=w2cs[:, :3, -1],
        in_ndc=False,
        image_size=image_shapes,
        device=device,
    )

    raster_settings = PointsRasterizationSettings(
        image_size=(H, W),
        radius=0.01,
        points_per_pixel=10,
        bin_size=100,
    )

    renderer = PointsZbufRenderer(
        rasterizer=PointsRasterizer(cameras=cameras, raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return render_pcd(pcd, imgs, masks, num_views, renderer, device, return_mask, return_depth)
