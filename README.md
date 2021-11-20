# TransMorph: Transformer for Medical Image Registration
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a> [![arXiv](https://img.shields.io/badge/arXiv-2104.06468-b31b1b.svg)](https://arxiv.org/abs/2104.06468)

keywords: Vision Transformer, Swin Transformer, convolutional neural networks, image registration

This is a **PyTorch** implementation of my paper:

<a href="https://arxiv.org/abs/2104.06468">Chen, Junyu, et al. "TransMorph: Transformer for Medical Image Registratio. " arXiv, 2021.</a>
## TransMorph Variants:
There are **four** TransMorph variants: *TransMorph*, *TransMorph-diff*, *TransMorph-bspl*, and *TransMorph-Bayes*.

1. ***TransMorph:*** A hybrid Transformer-ConvNet network for image registration.
2. ***TransMorph-diff:*** A probabilistic TransMorph that ensures a diffeomorphism.
3. ***TransMorph-bspl:*** A B-spline TransMorph that ensures a diffeomorphism.
4. ***TransMorph-Bayes:*** A Bayesian uncerntainty TransMorph that produces registration uncertainty estimate.

***train_xxx.py*** and ***infer_xxx.py*** are the training and inference scripts for TransMorph variants.

***model:*** This folder contains all the models.

***Dataset:*** Due to restrictions, we cannot distribute our brain MRI data. However, several brain MRI datasets are publicly available online: IXI, ADNI, OASIS, ABIDE, etc. Note that those datasets may not contain labels (segmentation). To generate labels, you can use FreeSurfer, which is an open-source software for normalizing brain MRI images. Here are some useful commands in FreeSurfer: <a href="https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/PreprocessingMRI.md">Brain MRI preprocessing and subcortical segmentation using FreeSurfer</a>.

## Model Architecture:
<img src="https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/example_imgs/architecture.jpg" width="800"/>

## Example Results:
<img src="https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/example_imgs/Results.jpg" width="1000"/>

## Quantitative Results:
### Inter-patient Brain MRI:
<img src="https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/example_imgs/brain_dsc.jpg" width="900"/>

### XCAT-to-CT:
<img src="https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/example_imgs/ct_dsc.jpg" width="900"/>

## Reference:
<a href="https://github.com/microsoft/Swin-Transformer">Swin Transformer</a>\
<a href="https://github.com/uncbiag/easyreg">easyreg</a>\
<a href="https://github.com/qiuhuaqi/midir">MIDIR</a>\
<a href="https://github.com/voxelmorph/voxelmorph">VoxelMorph</a>
### <a href="https://junyuchen245.github.io"> About Me</a>
