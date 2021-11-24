'''
Pyramid vision transformer for Image Registration

Paper:
Chen, J., Du, Y., He, Y., Segars, P. W., Li, Y., & Frey, E. C. (2021).
TransMorph: Transformer for Unsupervised Medical Image Registration. arXiv preprint arXiv:2111.10480.

Original PVT code was retrieved from:
https://github.com/whai362/PVT

Original PVT paper:
Wang, W., Xie, E., Li, X., Fan, D. P., Song, K., Liang, D., ... & Shao, L. (2021).
Pyramid vision transformer: A versatile backbone for dense prediction without convolutions.
arXiv preprint arXiv:2102.12122.

Junyu Chen
jchen245@jhmi.edu
Johns Hopkins University
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_, to_3tuple
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import models.configs_PVT as configs
from torch.distributions.normal import Normal
import torch.nn.functional as nnf
import sys

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv3d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W, L):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W, L)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W, L):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W, L))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=(160, 192, 224), patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_3tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0 and img_size[2] % patch_size[2] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W, self.L = img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2]
        self.num_patches = self.H * self.W * self.L
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W, L = x.shape

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W, L = H // self.patch_size[0], W // self.patch_size[1], L // self.patch_size[2]

        return x, (H, W, L)


class PyramidVisionTransformer(nn.Module):
    def __init__(self, img_size=(160, 192, 224), patch_size=16, in_chans=2, num_classes=1000, embed_dims=(64, 128, 256, 512),
                 num_heads=(1, 2, 4, 8), mlp_ratios=(4, 4, 4, 4), qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=(3, 4, 6, 3), sr_ratios=(8, 4, 2, 1), F4=False):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.F4 = F4

        # patch_embed
        self.patch_embed1 = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                       embed_dim=embed_dims[0])
        self.patch_embed2 = PatchEmbed(img_size=(img_size[0] // 4, img_size[1] // 4, img_size[2] // 4), patch_size=2, in_chans=embed_dims[0],
                                       embed_dim=embed_dims[1])
        self.patch_embed3 = PatchEmbed(img_size=(img_size[0] // 8, img_size[1] // 8, img_size[2] // 8), patch_size=2, in_chans=embed_dims[1],
                                       embed_dim=embed_dims[2])
        self.patch_embed4 = PatchEmbed(img_size=(img_size[0] // 16, img_size[1] // 16, img_size[2] // 16), patch_size=2, in_chans=embed_dims[2],
                                       embed_dim=embed_dims[3])

        # pos_embed
        self.pos_embed1 = nn.Parameter(torch.zeros(1, self.patch_embed1.num_patches, embed_dims[0]))
        self.pos_drop1 = nn.Dropout(p=drop_rate)
        self.pos_embed2 = nn.Parameter(torch.zeros(1, self.patch_embed2.num_patches, embed_dims[1]))
        self.pos_drop2 = nn.Dropout(p=drop_rate)
        self.pos_embed3 = nn.Parameter(torch.zeros(1, self.patch_embed3.num_patches, embed_dims[2]))
        self.pos_drop3 = nn.Dropout(p=drop_rate)
        self.pos_embed4 = nn.Parameter(torch.zeros(1, self.patch_embed4.num_patches + 1, embed_dims[3]))
        self.pos_drop4 = nn.Dropout(p=drop_rate)

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])

        # init weights
        trunc_normal_(self.pos_embed1, std=.02)
        trunc_normal_(self.pos_embed2, std=.02)
        trunc_normal_(self.pos_embed3, std=.02)
        trunc_normal_(self.pos_embed4, std=.02)
        self.apply(self._init_weights)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _get_pos_embed(self, pos_embed, patch_embed, H, W, L):
        if H * W * L == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, patch_embed.L, -1).permute(0, 4, 1, 2, 3),
                size=(H, W, L), mode="trilinear").reshape(1, -1, H * W * L).permute(0, 2, 1)

    def forward_features(self, x):
        outs = []

        B = x.shape[0]

        # stage 1
        x, (H, W, L) = self.patch_embed1(x)
        pos_embed1 = self._get_pos_embed(self.pos_embed1, self.patch_embed1, H, W, L)
        x = x + pos_embed1
        x = self.pos_drop1(x)
        for blk in self.block1:
            x = blk(x, H, W, L)
        x = x.reshape(B, H, W, L, -1).permute(0, 4, 1, 2, 3).contiguous()
        outs.append(x)

        # stage 2
        x, (H, W, L) = self.patch_embed2(x)
        pos_embed2 = self._get_pos_embed(self.pos_embed2, self.patch_embed2, H, W, L)
        x = x + pos_embed2
        x = self.pos_drop2(x)
        for blk in self.block2:
            x = blk(x, H, W, L)
        x = x.reshape(B, H, W, L, -1).permute(0, 4, 1, 2, 3).contiguous()
        outs.append(x)

        # stage 3
        x, (H, W, L) = self.patch_embed3(x)
        pos_embed3 = self._get_pos_embed(self.pos_embed3, self.patch_embed3, H, W, L)
        x = x + pos_embed3
        x = self.pos_drop3(x)
        for blk in self.block3:
            x = blk(x, H, W, L)
        x = x.reshape(B, H, W, L, -1).permute(0, 4, 1, 2, 3).contiguous()
        outs.append(x)

        # stage 4
        x, (H, W, L) = self.patch_embed4(x)
        pos_embed4 = self._get_pos_embed(self.pos_embed4[:, 1:], self.patch_embed4, H, W, L)
        x = x + pos_embed4
        x = self.pos_drop4(x)
        for blk in self.block4:
            x = blk(x, H, W, L)
        x = x.reshape(B, H, W, L, -1).permute(0, 4, 1, 2, 3).contiguous()
        outs.append(x)

        return outs

    def forward(self, x):
        x = self.forward_features(x)

        if self.F4:
            x = x[3:4]

        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict

class Conv3dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        relu = nn.LeakyReLU(inplace=True)
        if not use_batchnorm:
            nm = nn.InstanceNorm3d(out_channels)
        else:
            nm = nn.BatchNorm3d(out_channels)

        super(Conv3dReLU, self).__init__(conv, nm, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv3dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv3dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class RegistrationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        conv3d.weight = nn.Parameter(Normal(0, 1e-5).sample(conv3d.weight.shape))
        conv3d.bias = nn.Parameter(torch.zeros(conv3d.bias.shape))
        super().__init__(conv3d)

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer

    Obtained from https://github.com/voxelmorph/voxelmorph
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

class PVTVNetSkip(nn.Module):
    def __init__(self, config):
        super(PVTVNetSkip, self).__init__()
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dims = config.embed_dims
        self.transformer = PyramidVisionTransformer(img_size=config.img_size,
                                                    patch_size=config.patch_size,
                                                    embed_dims=config.embed_dims,
                                                    depths=config.depths,
                                                    num_heads=config.num_heads,
                                                    mlp_ratios=config.mlp_ratios,
                                                    qkv_bias=config.qkv_bias,
                                                    drop_rate=config.drop_rate,
                                                    drop_path_rate=config.drop_path_rate,
                                                    sr_ratios=config.sr_ratios,)
        self.up1 = DecoderBlock(embed_dims[-1], embed_dims[-2], skip_channels=embed_dims[-2] if if_transskip else 0, use_batchnorm=False)  # 384, 20, 20, 64
        self.up2 = DecoderBlock(embed_dims[-2], embed_dims[-3], skip_channels=embed_dims[-3] if if_transskip else 0, use_batchnorm=False)  # 384, 40, 40, 64
        self.up3 = DecoderBlock(embed_dims[-3], embed_dims[-4], skip_channels=embed_dims[-4] if if_convskip else 0, use_batchnorm=False)  # 384, 80, 80, 128
        self.up4 = DecoderBlock(embed_dims[-4], 16, skip_channels=16 if if_convskip else 0, use_batchnorm=False)  # 384, 160, 160, 256
        self.up5 = DecoderBlock(16, config.reg_head_chan, skip_channels=config.reg_head_chan if if_convskip else 0, use_batchnorm=False)  # 384, 160, 160, 256
        self.c1 = Conv3dReLU(2, config.reg_head_chan, 3, 1, use_batchnorm=False)
        self.c2 = Conv3dReLU(2, config.reg_head_chan, 3, 1, use_batchnorm=False)
        self.reg_head = RegistrationHead(
            in_channels=config.reg_head_chan,
            out_channels=3,
            kernel_size=3,
        )
        self.spatial_trans = SpatialTransformer(config.img_size)
        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)

    def forward(self, x):
        source = x[:, 0:1, :, :]
        if self.if_convskip:
            x_s0 = x.clone()
            x_s1 = self.avg_pool(x)
            f4 = self.c1(x_s1)
            f5 = self.c2(x_s0)
        else:
            f4 = None
            f5 = None
        out = self.transformer(x)
        if self.if_transskip:
            f1 = out[-2]
            f2 = out[-3]
            f3 = out[-4]
        else:
            f1 = None
            f2 = None
            f3 = None
        x = self.up1(out[-1], f1)
        x = self.up2(x, f2)
        x = self.up3(x, f3)
        x = self.up4(x, f4)
        x = self.up5(x, f5)
        flow = self.reg_head(x)
        out = self.spatial_trans(source, flow)
        return out, flow


CONFIGS = {
    'PVT-Net': configs.get_3DPVTNet_config(),
}