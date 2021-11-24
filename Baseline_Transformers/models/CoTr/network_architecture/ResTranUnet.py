'''
VoxelMorph

Original code retrieved from:
https://github.com/YtongXie/CoTr

Original paper:
Xie, Y., Zhang, J., Shen, C., & Xia, Y. (2021).
CoTr: Efficiently Bridging CNN and Transformer for 3D Medical Image Segmentation.
arXiv preprint arXiv:2103.03024.

Modified and tested by:
Junyu Chen
jchen245@jhmi.edu
Johns Hopkins University
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.CoTr.network_architecture import CNNBackbone
from models.CoTr.network_architecture.neural_network import SegmentationNetwork
from models.CoTr.network_architecture.DeTrans.DeformableTrans import DeformableTransformer
from models.CoTr.network_architecture.DeTrans.position_encoding import build_position_encoding
from torch.distributions.normal import Normal

class Conv3d_wd(nn.Conv3d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1,1), padding=(0,0,0), dilation=(1,1,1), groups=1, bias=False):
        super(Conv3d_wd, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True).mean(dim=4, keepdim=True)
        weight = weight - weight_mean
        # std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1, 1) + 1e-5
        std = torch.sqrt(torch.var(weight.view(weight.size(0), -1), dim=1) + 1e-12).view(-1, 1, 1, 1, 1)
        weight = weight / std.expand_as(weight)
        return F.conv3d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def conv3x3x3(in_planes, out_planes, kernel_size, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), groups=1, bias=False, weight_std=False):
    "3x3x3 convolution with padding"
    if weight_std:
        return Conv3d_wd(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
    else:
        return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)


def Norm_layer(norm_cfg, inplanes):
    out = None
    if norm_cfg == 'BN':
        out = nn.BatchNorm3d(inplanes)
    elif norm_cfg == 'SyncBN':
        out = nn.SyncBatchNorm(inplanes)
    elif norm_cfg == 'GN':
        out = nn.GroupNorm(16, inplanes)
    else: # norm_cfg == 'IN':
        out = nn.InstanceNorm3d(inplanes, affine=True)
    return out


def Activation_layer(activation_cfg, inplace=True):
    out = None
    if activation_cfg == 'ReLU':
        out = nn.ReLU(inplace=inplace)
    else: # activation_cfg == 'LeakyReLU':
        out = nn.LeakyReLU(negative_slope=1e-2, inplace=inplace)

    return out


class Conv3dBlock(nn.Module):
    def __init__(self,in_channels,out_channels,norm_cfg,activation_cfg,kernel_size,stride=(1, 1, 1),padding=(0, 0, 0),dilation=(1, 1, 1),bias=False,weight_std=False):
        super(Conv3dBlock,self).__init__()
        self.conv = conv3x3x3(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, weight_std=weight_std)
        self.norm = Norm_layer(norm_cfg, out_channels)
        self.nonlin = Activation_layer(activation_cfg, inplace=True)
    def forward(self,x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.nonlin(x)
        return x

class ResBlock(nn.Module):

    def __init__(self, inplanes, planes, norm_cfg, activation_cfg, weight_std=False):
        super(ResBlock, self).__init__()
        self.resconv1 = Conv3dBlock(inplanes, planes, norm_cfg, activation_cfg, kernel_size=3, stride=1, padding=1, bias=False, weight_std=weight_std)
        self.resconv2 = Conv3dBlock(planes, planes, norm_cfg, activation_cfg, kernel_size=3, stride=1, padding=1, bias=False, weight_std=weight_std)

    def forward(self, x):
        residual = x

        out = self.resconv1(x)
        out = self.resconv2(out)
        out = out + residual

        return out

class U_ResTran3D(nn.Module):
    def __init__(self, norm_cfg='BN', activation_cfg='ReLU', img_size=None, weight_std=False):
        super(U_ResTran3D, self).__init__()

        self.upsamplex2 = nn.Upsample(scale_factor=(1,2,2), mode='trilinear')

        self.transposeconv_stage2 = nn.ConvTranspose3d(384, 384, kernel_size=(2,2,2), stride=(2,2,2), bias=False)
        self.transposeconv_stage1 = nn.ConvTranspose3d(384, 192, kernel_size=(2,2,2), stride=(2,2,2), bias=False)
        self.transposeconv_stage0 = nn.ConvTranspose3d(192, 64, kernel_size=(2,2,2), stride=(2,2,2), bias=False)

        self.stage2_de = ResBlock(384, 384, norm_cfg, activation_cfg, weight_std=weight_std)
        self.stage1_de = ResBlock(192, 192, norm_cfg, activation_cfg, weight_std=weight_std)
        self.stage0_de = ResBlock(64, 64, norm_cfg, activation_cfg, weight_std=weight_std)

        self.ds2_cls_conv = RegistrationHead(in_channels=384,out_channels=3,kernel_size=3,)
        self.ds1_cls_conv = RegistrationHead(in_channels=192,out_channels=3,kernel_size=3,)
        self.ds0_cls_conv = RegistrationHead(in_channels=64,out_channels=3,kernel_size=3,)

        self.cls_conv = RegistrationHead(in_channels=64,out_channels=3,kernel_size=3,)
        self.spatial_trans = SpatialTransformer(img_size)
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, Conv3d_wd, nn.ConvTranspose3d)):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm3d, nn.SyncBatchNorm, nn.InstanceNorm3d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.backbone = CNNBackbone.Backbone(in_channels=2, depth=9, norm_cfg=norm_cfg, activation_cfg=activation_cfg, weight_std=weight_std)
        total = sum([param.nelement() for param in self.backbone.parameters()])
        print('  + Number of Backbone Params: %.2f(e6)' % (total / 1e6))

        self.position_embed = build_position_encoding(mode='v2', hidden_dim=384)
        self.encoder_Detrans = DeformableTransformer(d_model=384, dim_feedforward=1536, dropout=0.1, activation='gelu', num_feature_levels=2, nhead=4, num_encoder_layers=4, enc_n_points=4)
        total = sum([param.nelement() for param in self.encoder_Detrans.parameters()])
        print('  + Number of Transformer Params: %.2f(e6)' % (total / 1e6))

    def posi_mask(self, x):

        x_fea = []
        x_posemb = []
        masks = []
        for lvl, fea in enumerate(x):
            if lvl > 1:
                x_fea.append(fea)
                x_posemb.append(self.position_embed(fea))
                masks.append(torch.zeros((fea.shape[0], fea.shape[2], fea.shape[3], fea.shape[4]), dtype=torch.bool).cuda())

        return x_fea, masks, x_posemb


    def forward(self, inputs):
        # # %%%%%%%%%%%%% CoTr
        source = inputs[:, 0:1, ...]
        x_convs = self.backbone(inputs)
        x_fea, masks, x_posemb = self.posi_mask(x_convs)
        x_trans = self.encoder_Detrans(x_fea, masks, x_posemb)

        # # Single_scale
        # # x = self.transposeconv_stage2(x_trans.transpose(-1, -2).view(x_convs[-1].shape))
        # # skip2 = x_convs[-2]
        # Multi-scale
        x = self.transposeconv_stage2(x_trans[:, -3360:,].transpose(-1, -2).reshape(x_convs[-1].shape))
        skip2 = x_trans[:, 0:26880].transpose(-1, -2).view(x_convs[-2].shape)

        x = x + skip2
        x = self.stage2_de(x)
        #ds2 = self.ds2_cls_conv(x)

        x = self.transposeconv_stage1(x)
        skip1 = x_convs[-3]
        x = x + skip1
        x = self.stage1_de(x)
        #ds1 = self.ds1_cls_conv(x)

        x = self.transposeconv_stage0(x)
        skip0 = x_convs[-4]
        x = x + skip0
        x = self.stage0_de(x)
        #ds0 = self.ds0_cls_conv(x)


        result = self.upsamplex2(x)
        flow = self.cls_conv(result)
        out = self.spatial_trans(source, flow)
        return out, flow


class ResTranUnet(SegmentationNetwork):
    """
    ResTran-3D Unet
    """
    def __init__(self, norm_cfg='BN', activation_cfg='ReLU', img_size=(160, 192, 224), num_classes=None, weight_std=False, deep_supervision=False):
        super().__init__()
        self.do_ds = False
        self.U_ResTran3D = U_ResTran3D(norm_cfg, activation_cfg, img_size, weight_std) # U_ResTran3D
        if weight_std==False:
            self.conv_op = nn.Conv3d
        else:
            self.conv_op = Conv3d_wd
        if norm_cfg=='BN':
            self.norm_op = nn.BatchNorm3d
        if norm_cfg=='SyncBN':
            self.norm_op = nn.SyncBatchNorm
        if norm_cfg=='GN':
            self.norm_op = nn.GroupNorm
        if norm_cfg=='IN':
            self.norm_op = nn.InstanceNorm3d
        self.dropout_op = nn.Dropout3d
        self.num_classes = num_classes
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        total = sum([param.nelement() for param in self.U_ResTran3D.parameters()])
        print('  + Number of Total Params: %.2f(e6)' % (total / 1e6))
    def forward(self, x):
        seg_output = self.U_ResTran3D(x)
        if self._deep_supervision and self.do_ds:
            return seg_output
        else:
            return seg_output

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

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)
