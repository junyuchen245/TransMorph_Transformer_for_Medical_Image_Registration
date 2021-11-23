'''
B-Spline TransMorph with Diffeomorphism

Paper:
Chen, J., Du, Y., He, Y., Segars, P. W., Li, Y., & Frey, E. C. (2021). 
TransMorph: Transformer for Unsupervised Medical Image Registration. arXiv preprint.

Base code for B-Spline registration was obtained from: https://github.com/qiuhuaqi/midir

Original paper for learning-based B-Spline registration:
Qiu, H., Qin, C., Schuh, A., Hammernik, K., & Rueckert, D. (2021, February). 
Learning Diffeomorphic and Modality-invariant Registration using B-splines. 
In Medical Imaging with Deep Learning.

Junyu Chen
jchen245@jhmi.edu
Johns Hopkins University
'''

import math
import torch
from torch import nn as nn
from torch.nn import functional as F
import models.transformation as transformation
import models.TransMorph as TM
import models.configs_TransMorph_bspl as configs

def convNd(ndim,
           in_channels,
           out_channels,
           kernel_size=3,
           stride=1,
           padding=1,
           a=0.):
    """
    Convolution of generic dimension
    Args:
        in_channels: (int) number of input channels
        out_channels: (int) number of output channels
        kernel_size: (int) size of the convolution kernel
        stride: (int) convolution stride (step size)
        padding: (int) outer padding
        ndim: (int) model dimension
        a: (float) leaky-relu negative slope for He initialisation
    Returns:
        (nn.Module instance) Instance of convolution module of the specified dimension
    """
    conv_nd = getattr(nn, f"Conv{ndim}d")(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding)
    nn.init.kaiming_uniform_(conv_nd.weight, a=a)
    return conv_nd


def interpolate_(x, scale_factor=None, size=None, mode=None):
    """ Wrapper for torch.nn.functional.interpolate """
    if mode == 'nearest':
        mode = mode
    else:
        ndim = x.ndim - 2
        if ndim == 1:
            mode = 'linear'
        elif ndim == 2:
            mode = 'bilinear'
        elif ndim == 3:
            mode = 'trilinear'
        else:
            raise ValueError(f'Data dimension ({ndim}) must be 2 or 3')
    y = F.interpolate(x,
                      scale_factor=scale_factor,
                      size=size,
                      mode=mode,
                      )
    return y

class TranMorphBSplineNet(nn.Module):
    def __init__(self,
                 config,
                 ):
        """
        Network to parameterise Cubic B-spline transformation
        """
        super(TranMorphBSplineNet, self).__init__()

        # determine and set output control point sizes from image size and control point spacing
        ndim = 3
        img_size = config.img_size
        cps = config.cps
        resize_channels = config.resize_channels
        for i, c in enumerate(cps):
            if c > 8 or c < 2:
                raise ValueError(f"Control point spacing ({c}) at dim ({i}) not supported, must be within [1, 8]")
        self.output_size = tuple([int(math.ceil((imsz-1) / c) + 1 + 2)
                                  for imsz, c in zip(img_size, cps)])

        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.transformer = TM.SwinTransformer(patch_size=config.patch_size,
                                              in_chans=config.in_chans,
                                              embed_dim=config.embed_dim,
                                              depths=config.depths,
                                              num_heads=config.num_heads,
                                              window_size=config.window_size,
                                              mlp_ratio=config.mlp_ratio,
                                              qkv_bias=config.qkv_bias,
                                              drop_rate=config.drop_rate,
                                              drop_path_rate=config.drop_path_rate,
                                              ape=config.ape,
                                              spe=config.spe,
                                              patch_norm=config.patch_norm,
                                              use_checkpoint=config.use_checkpoint,
                                              out_indices=config.out_indices,
                                              pat_merg_rf=config.pat_merg_rf,)
        self.up0 = TM.DecoderBlock(embed_dim * 8, embed_dim * 4, skip_channels=embed_dim * 4 if if_transskip else 0,
                                   use_batchnorm=False)
        self.up1 = TM.DecoderBlock(embed_dim * 4, embed_dim * 2, skip_channels=embed_dim * 2 if if_transskip else 0,
                                   use_batchnorm=False)  # 384, 20, 20, 64
        self.up2 = TM.DecoderBlock(embed_dim * 2, embed_dim, skip_channels=embed_dim if if_transskip else 0,
                                   use_batchnorm=False)  # 384, 40, 40, 64
        self.up3 = TM.DecoderBlock(embed_dim, embed_dim // 2, skip_channels=embed_dim // 2 if if_convskip else 0,
                                   use_batchnorm=False)  # 384, 80, 80, 128
        self.c1 = TM.Conv3dReLU(2, embed_dim // 2, 3, 1, use_batchnorm=False)

        # conv layers following resizing
        self.resize_conv = nn.ModuleList()
        for i in range(len(resize_channels)):
            if i == 0:
                in_ch = embed_dim // 2
            else:
                in_ch = resize_channels[i-1]
            out_ch = resize_channels[i]
            self.resize_conv.append(nn.Sequential(convNd(ndim, in_ch, out_ch, a=0.2),
                                                  nn.LeakyReLU(0.2)))

        # final prediction layer
        self.out_layer = convNd(ndim, resize_channels[-1], ndim)
        self.transform = transformation.CubicBSplineFFDTransform(ndim=3, svf=True, cps=cps)

    def forward(self, inputs):
        src, tar = inputs
        x = torch.cat((src, tar), dim=1)
        if self.if_convskip:
            x_s1 = self.avg_pool(x)
            f4 = self.c1(x_s1)
        else:
            f4 = None
        out = self.transformer(x)

        if self.if_transskip:
            f1 = out[-2]
            f2 = out[-3]
            f3 = out[-4]
        else:
            f1 = None
            f2 = None
            f3 = None
        x = self.up0(out[-1], f1)
        x = self.up1(x, f2)
        x = self.up2(x, f3)
        dec_out = self.up3(x, f4)

        # resize output of encoder-decoder
        x = interpolate_(dec_out, size=self.output_size)

        # layers after resize
        for resize_layer in self.resize_conv:
            x = resize_layer(x)
        x = self.out_layer(x)
        flow, disp = self.transform(x)
        y = transformation.warp(src, disp)
        return y, flow, disp

CONFIGS = {
    'TransMorphBSpline': configs.get_TransMorphBspl_config(),
}
