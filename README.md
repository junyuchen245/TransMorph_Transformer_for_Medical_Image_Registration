# TransMorph: Transformer for Medical Image Registration
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a> [![arXiv](https://img.shields.io/badge/arXiv-2104.06468-b31b1b.svg)](https://arxiv.org/abs/2104.06468)

keywords: Vision Transformer, Swin Transformer, convolutional neural networks, image registration

This is a **PyTorch** implementation of my paper:

<a href="https://arxiv.org/abs/2104.06468">Chen, Junyu, et al. "TransMorph: Transformer for Medical Image Registratio. " arXiv, 2021.</a>

## Model Architecture:
<img src="https://github.com/junyuchen245/ViT-V-Net_for_3D_Image_Registration/blob/main/figures/net_arch.jpg" width="700"/>


## Flops counting for JHU MRI:
Model           | Input Resolution | Params(M) | Flops/MACs(G)| DSC          |
---             |---               |---        |---           |---           |
VoxelMoprh-1    |1x2x160x192x224   | 0.27      |305.07        |0.706 +- 0.137|
VoxelMoprh-2    |1x2x160x192x224   | 0.30      |400.06        |0.711 +- 0.135|
VoxelMoprhx2    |1x2x160x192x224   | 1.2       |1554.61       |0.726 +- 0.130|
VoxelMorph-diff |1x2x160x192x224   | 0.30      |91.71         |0.701 +- 0.139|
VoxelMoprh-huge |1x2x160x192x224   | 63.25     |3656.19       |0.732 +- 0.127|
CycleMorph      |1x2x160x192x224   | 0.72      |966.9         |0.694 +- 0.138|
BSplineNet      |1x2x160x192x224   | 0.27      |47.38         |0.700 +- 0.135|
ICNet           |1x2x160x192x224   | 1.11      |215.12        |0.648 +- 0.149|
SYMNet          |1x2x160x192x224   | 1.12      |297.78        |0.673 +- 0.145|
PVTNet-V0       |1x2x160x192x224   | 7.24      |195.98        |0.712 +- 0.134|
PVTNet-V1       |1x2x160x192x224   | 61.84     |246.67        |0.718 +- 0.133|
CoTr            |1x2x160x192x224   | 38.74     |2275.16       |0.714 +- 0.134|
ViT-V-Net       |1x2x160x192x224   | 110.62    |406.44        |0.725 +- 0.130|
SwinNet-v0      |1x2x160x192x224   | 63.63     |686.75        |0.733 +- 0.128|
SwinNet-v0-diff |1x2x160x192x224   | 63.42     |280.98        |0.718 +- 0.131|
SwinNet-v0-SYM  |1x2x160x192x224   | 63.63     |686.75        |
SwinNet-v0-BSpl |1x2x160x192x224   | 63.67     |454.41        |0.730 +- 0.127|
