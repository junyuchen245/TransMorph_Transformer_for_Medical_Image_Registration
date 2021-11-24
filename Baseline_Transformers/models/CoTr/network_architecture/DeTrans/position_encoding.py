"""
Positional encodings for the transformer.
"""
import math
import torch
from torch import nn
from typing import Optional
from torch import Tensor

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=[64, 64, 64], temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        bs, c, d, h, w = x.shape
        mask = torch.zeros(bs, d, h, w, dtype=torch.bool).cuda()
        assert mask is not None
        not_mask = ~mask
        d_embed = not_mask.cumsum(1, dtype=torch.float32)
        y_embed = not_mask.cumsum(2, dtype=torch.float32)
        x_embed = not_mask.cumsum(3, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            d_embed = (d_embed - 0.5) / (d_embed[:, -1:, :, :] + eps) * self.scale
            y_embed = (y_embed - 0.5) / (y_embed[:, :, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, :, -1:] + eps) * self.scale

        dim_tx = torch.arange(self.num_pos_feats[0], dtype=torch.float32, device=x.device)
        dim_tx = self.temperature ** (3 * (dim_tx // 3) / self.num_pos_feats[0])

        dim_ty = torch.arange(self.num_pos_feats[1], dtype=torch.float32, device=x.device)
        dim_ty = self.temperature ** (3 * (dim_ty // 3) / self.num_pos_feats[1])

        dim_td = torch.arange(self.num_pos_feats[2], dtype=torch.float32, device=x.device)
        dim_td = self.temperature ** (3 * (dim_td // 3) / self.num_pos_feats[2])

        pos_x = x_embed[:, :, :, :, None] / dim_tx
        pos_y = y_embed[:, :, :, :, None] / dim_ty
        pos_d = d_embed[:, :, :, :, None] / dim_td

        pos_x = torch.stack((pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos_y = torch.stack((pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos_d = torch.stack((pos_d[:, :, :, :, 0::2].sin(), pos_d[:, :, :, :, 1::2].cos()), dim=5).flatten(4)

        pos = torch.cat((pos_d, pos_y, pos_x), dim=4).permute(0, 4, 1, 2, 3)
        return pos


def build_position_encoding(mode, hidden_dim):
    N_steps = hidden_dim // 3
    if (hidden_dim % 3) != 0:
        N_steps = [N_steps, N_steps, N_steps + hidden_dim % 3]
    else:
        N_steps = [N_steps, N_steps, N_steps]
    
    if mode in ('v2', 'sine'):
        position_embedding = PositionEmbeddingSine(num_pos_feats=N_steps, normalize=True)
    else:
        raise ValueError(f"not supported {mode}")

    return position_embedding
