import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config

class DiffMaskPredictor(nn.Module):
    def __init__(self, input_dim=4608, patch_grid=(10, 15, 189), output_grid=(81, 480, 720), hidden_dim=256):
        """
        Args:
            input_dim (int): concatenated feature dimension, e.g. 1536 * num_selected_layers
            patch_grid (tuple): (F_p, H_p, W_p) - patch token grid shape (e.g., from transformer block)
            output_grid (tuple): (F, H, W) - final full resolution shape for mask
            hidden_dim (int): intermediate conv/linear hidden dim
        """
        super().__init__()
        self.F_p, self.H_p, self.W_p = patch_grid
        self.F, self.H, self.W = output_grid

        self.project = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): shape [B, L, D_total], L = F_p  H_p  W_p
        Returns:
            Tensor: predicted diff mask, shape [B, 1, F, H, W]
        """
        B, L, D = x.shape
        assert L == self.F_p * self.H_p * self.W_p, \
            f"Input token length {L} doesn't match patch grid ({self.F_p}, {self.H_p}, {self.W_p})"

        x = self.project(x)                         # [B, L, 1]
        x = x.view(B, 1, self.F_p, self.H_p, self.W_p)  # [B, 1, F_p, H_p, W_p]
        x = F.interpolate(
            x, size=(self.F, self.H, self.W),
            mode="trilinear", align_corners=False   # upsample to match ground truth resolution
        )
        return x  # [B, 1, F, H, W]
