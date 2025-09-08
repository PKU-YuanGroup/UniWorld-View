import torch
from pytorch3d.renderer import DirectionalLights, MeshRenderer
from pytorch3d.structures import Meshes


def create_lights(device, R, ambient_color=0.4, diffuse_color=0.3, specular_color=0.2):
    R = R.mT.to(device)
    light_direction = -R[:, :, 2]  # The direction will be set to look in the same direction as the camera
    lights = DirectionalLights(
        direction=light_direction,  # The direction is set as the negative z-axis from camera's perspective
        ambient_color=torch.tensor([[ambient_color, ambient_color, ambient_color]]).to(device),
        diffuse_color=torch.tensor([[diffuse_color, diffuse_color, diffuse_color]]).to(device),
        specular_color=torch.tensor([[specular_color, specular_color, specular_color]]).to(device),
        device=device,
    )
    return lights

class MeshRendererWrap(MeshRenderer):
    """
    A class for rendering a batch of heterogeneous meshes. The class should
    be initialized with a rasterizer (a MeshRasterizer or a MeshRasterizerOpenGL)
    and shader class which each have a forward function.
    """

    def __init__(self, rasterizer, shader) -> None:
        super().__init__(rasterizer, shader)

    def forward(self, meshes_world: Meshes, **kwargs) -> [torch.Tensor, torch.Tensor]:
        """
        Render a batch of images from a batch of meshes by rasterizing and then
        shading.

        NOTE: If the blur radius for rasterization is > 0.0, some pixels can
        have one or more barycentric coordinates lying outside the range [0, 1].
        For a pixel with out of bounds barycentric coordinates with respect to a
        face f, clipping is required before interpolating the texture uv
        coordinates and z buffer so that the colors and depths are limited to
        the range for the corresponding face.
        For this set rasterizer.raster_settings.clip_barycentric_coords=True
        """
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)

        return images, fragments