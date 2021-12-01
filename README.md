# TransMorph: Transformer for Unsupervised Medical Image Registration
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a> [![arXiv](https://img.shields.io/badge/arXiv-2111.10480-b31b1b.svg)](https://arxiv.org/abs/2111.10480)

keywords: Vision Transformer, Swin Transformer, convolutional neural networks, image registration

This is a **PyTorch** implementation of my paper:

<a href="https://arxiv.org/abs/2111.10480">Chen, Junyu, et al. "TransMorph: Transformer for Unsupervised Medical Image Registration. " arXiv, 2021.</a>

## TransMorph DIR Variants:
There are **four** TransMorph variants: *TransMorph*, *TransMorph-diff*, *TransMorph-bspl*, and *TransMorph-Bayes*. \
Training and inference scripts are in `TransMorph/`, and the models are contained in `TransMorph/model/`.
1. ***TransMorph:*** A hybrid Transformer-ConvNet network for image registration.
2. ***TransMorph-diff:*** A probabilistic TransMorph that ensures a diffeomorphism.
3. ***TransMorph-bspl:*** A B-spline TransMorph that ensures a diffeomorphism.
4. ***TransMorph-Bayes:*** A Bayesian uncerntainty TransMorph that produces registration uncertainty estimate.

## TransMorph Affine Model:
The scripts for ***TransMorph affine*** model are in `TransMorph_affine/` folder.

`train_xxx.py` and `infer_xxx.py` are the training and inference scripts for TransMorph models.

## Loss Functions:
TransMorph supports both mono- and multi-modal registration. We provided the following loss functions for **image similarity** measurements (the links will take you to the code):
1. Mean squared error ([MSE](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html))
2. Normalized cross correlation ([NCC](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/205b09e8852ee4e415c36613413bc0bf3990f1f1/TransMorph/losses.py#L211))
3. Structural similarity index ([SSIM](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/f85635578d76e6076a262cd746a37d39c363a58d/TransMorph/losses.py#L103))
4. Mutual information ([MI](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/205b09e8852ee4e415c36613413bc0bf3990f1f1/TransMorph/losses.py#L338))
5. Local mutual information ([LMI](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/205b09e8852ee4e415c36613413bc0bf3990f1f1/TransMorph/losses.py#L396))
6. Modality independent neighbourhood descriptor with self-similarity context ([MIND-SSC](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/205b09e8852ee4e415c36613413bc0bf3990f1f1/TransMorph/losses.py#L287))

and the following **deformation regularizers**:
1. [Diffusion](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/205b09e8852ee4e415c36613413bc0bf3990f1f1/TransMorph/losses.py#L523)
2. [L1](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/205b09e8852ee4e415c36613413bc0bf3990f1f1/TransMorph/losses.py#L523)
3. [Anisotropic diffusion](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/205b09e8852ee4e415c36613413bc0bf3990f1f1/TransMorph/losses.py#L550)
4. [Bending energy](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/205b09e8852ee4e415c36613413bc0bf3990f1f1/TransMorph/losses.py#L570)

## Baseline Models:
We compared TransMorph with **eight** baseline registration methods + **four** Transformer architectures.

***Baseline registration methods:***\
*Training and inference scripts are in* `Baseline_registration_models/`
1. <a href="https://github.com/ANTsX/ANTsPy">SyN (ATNsPy)</a>
2. <a href="http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftyReg">NiftyReg</a>
3. <a href="https://github.com/brianlee324/torch-lddmm">LDDMM</a>
4. <a href="https://github.com/mattiaspaul/deedsBCV">deedsBCV</a>
5. <a href="https://github.com/voxelmorph/voxelmorph">VoxelMorph-1 & -2</a>
6. <a href="https://github.com/boahK/MEDIA_CycleMorph">CycleMorph</a>
7. <a href="https://github.com/qiuhuaqi/midir">MIDIR</a>

***Baseline Transformer architectures:***\
*Training and inference scripts are in* `Baseline_Transformers/`
1. <a href="https://github.com/whai362/PVT">PVT</a>
2. <a href="https://github.com/282857341/nnFormer">nnFormer</a>
3. <a href="https://github.com/YtongXie/CoTr">CoTr</a>
4. <a href="https://github.com/junyuchen245/ViT-V-Net_for_3D_Image_Registration_Pytorch">ViT-V-Net</a>

## Dataset:
Due to restrictions, we cannot distribute our brain MRI and CT data. However, several brain MRI datasets are publicly available online: IXI, ADNI, OASIS, ABIDE, etc. Note that those datasets may not contain labels (segmentation). To generate labels, you can use FreeSurfer, which is an open-source software for normalizing brain MRI images. Here are some useful commands in FreeSurfer: <a href="https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/PreprocessingMRI.md">Brain MRI preprocessing and subcortical segmentation using FreeSurfer</a>.

## Citation:
If you find this code is useful in your research, please consider to cite:
    
    @misc{chen2021transmorph,
    title={TransMorph: Transformer for Medical Image Registration}, 
    author={Junyu Chen and Yufan He and Eric C. Frey and Ye Li and Yong Du},
    year={2021},
    eprint={2111.10480},
    archivePrefix={arXiv},
    primaryClass={eess.IV}
    }
    
## TransMorph Architecture:
<img src="https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/example_imgs/architecture.jpg" width="800"/>

## Example Results:
### Qualitative comparisons:
<img src="https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/example_imgs/Results.jpg" width="1000"/>

### Uncertainty Estimate by TransMorph-Bayes:
<img src="https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/example_imgs/Brain_uncertainty.jpg" width="700"/>

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
