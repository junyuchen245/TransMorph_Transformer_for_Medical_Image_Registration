import ml_collections
from functools import partial
import torch.nn as nn
'''
********************************************************
                   PVT Transformer
********************************************************
img_size (int | tuple(int)): Input image size. Default (160, 192, 224)
patch_size (int | tuple(int)): Patch size. Default: 4
embed_dims (tuple(int)): Embedding dimensions in each layer.
num_heads (int): Number of attention heads in different layers.
mlp_ratio (tuple(float)): Ratio of mlp hidden dim to embedding dim.
qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
depths (tuple(int)): Depth of each PVT Transformer layer.
sr_ratios (tuple(float)): Spatial-reduction ratio that reduces the size of image via Conv.
drop_rate (float): Dropout rate. Default: 0
drop_path_rate (float): Stochastic depth rate. Default: 0.1
if_transskip (bool): Enable skip connections from Transformer Blocks
if_convskip (bool): Enable skip connections from Convolutional Blocks
reg_head_chan (int): Number of channels in the registration head (i.e., the final convolutional layer) 
embed_dim (int): Patch embedding dimension. Default: 96
depths (tuple(int)): Depth of each Swin Transformer layer.
num_heads (tuple(int)): Number of attention heads in different layers.
window_size (tuple): Window size. Default: 7
'''
def get_3DPVTNet_config():
    config = ml_collections.ConfigDict()
    config.img_size = (160, 192, 224)
    config.patch_size = 4
    config.embed_dims = (20, 40, 200, 320) #Differ from original PVT
    config.num_heads = (2, 4, 8, 16)
    config.mlp_ratios = (8, 8, 4, 4)
    config.qkv_bias = True
    config.norm_layer = partial(nn.LayerNorm, eps=1e-6)
    config.depths = (3, 10, 60, 3) #Differ from original PVT
    config.sr_ratios = (8, 4, 2, 1)
    config.drop_rate = 0.0
    config.drop_path_rate = 0.1
    config.if_convskip = True
    config.if_transskip = True
    config.reg_head_chan = 16
    return config