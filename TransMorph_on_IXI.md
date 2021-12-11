# TransMorph on IXI dataset
This page contains a brief description of using TransMorph variants and the baseline models on the [IXI dataset](https://brain-development.org/ixi-dataset/) for **atlas-to-patient registration**.\
All training and inference scripts mentioned on this page are in [`IXI/ (Currently Not Available)`]().

## IXI Dataset:
:exclamation: We will release the preprocessed dataset once we finish all the experiments.\
:exclamation: 12/10/2021 - We are currently testing TransMorph and the baseline models on the IXI dataset for atlas-to-patient registration. Once we finish testing, we will make our preprocessed IXI dataset publicly available under the Creative Commons [CC BY-SA 3.0 license](https://creativecommons.org/licenses/by-sa/3.0/legalcode). If you use the data, you should acknowledge the TransMorph paper:

    @article{chen2021transmorph,
    title={TransMorph: Transformer for unsupervised medical image registration},
    author={Chen, Junyu and Du, Yong and He, Yufan and Segars, William P and Li, Ye and Frey, Eric C},
    journal={arXiv preprint arXiv:2111.10480},
    year={2021}
    }

and acknowledge the source of the IXI data: https://brain-development.org/ixi-dataset/ 

- ***Preprocessing:*** The IXI dataset was preprocessed (e.g., skull stripping, affine alignment, and subcortical segmentation) by using [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/fswiki). The preprocessing steps we used in FreeSurfer are listed here - <a href="https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/PreprocessingMRI.md">Brain MRI preprocessing and subcortical segmentation using FreeSurfer</a>
- ***Download:*** Click on the link to start downloading the preprocessed dataset - [Preprocessed IXI Dataset (Currently Not Available)]()
- ***Train-Val-Test split:*** There are **576** brain MRI volumes in total. We split the dataset into a ratio of **7:1:2**, where **403** for training (`IXI_data/Train/`), **58** for validation (`IXI_data/Val/`), and **115** for testing (`IXI_data/Test/`).
- ***Atlas image:*** Additionally, there is an atlas MRI volume and the corresponding subcortical segmentation (`IXI_data/altas.pkl`). This atlas volume was obtained from [CycleMorph](https://github.com/boahK/MEDIA_CycleMorph).
- ***File format:*** Each `.pkl` file contains a T1 weighted brain MRI and its corresponding subcortical segmentation. You can read `.pkl` file in python by doing:
```python
import pickle
def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

image, label = pkload("image.pkl")
```

## Pre-trained Model Weights
Click on the links to start downloading the pre-trained weights.
### TransMorph Variants:
1. [TransMorph](https://drive.google.com/uc?export=download&id=1SDWj2ppvmkXMn1qw8jFkAeQqW3B8VZcu)
2. [TransMorph-Bayes](https://drive.google.com/uc?export=download&id=1TxCFeUokywV5kff_A1EjrCY6QjH_jFgb)
3. [TransMorph-diff](https://drive.google.com/uc?export=download&id=1K_6-CS_x7tkgYQWXGMhGIhksk83pCBu4)
4. [TransMorph-bspl](https://drive.google.com/uc?export=download&id=1ZLNM9KUP8KtCXBLwXRc9dx3OdWft6eMY)

### Baseline Models:
***Pre-trained baseline registration models:***
1. [VoxelMorph-1](https://drive.google.com/uc?export=download&id=1pjujL0PTELYy3TS_nj0BFnJjBF7OUqqm)
2. [VoxelMorph-2](https://drive.google.com/uc?export=download&id=1awrgjTWCykjpMlBVUbvILBVpZTzBXd4V)
3. CycleMorph (Under testing)
4. [MIDIR](https://drive.google.com/uc?export=download&id=1JWCF1pqmF2FE8mc0MVP31y3KKQ08M-fM)

***Pre-trained baseline Transformer-based registration models:***
1. PVT (Under testing)
2. nnFormer (Under testing)
3. CoTr (Under testing)
4. ViT-V-Net (Under testing)

## Instructions on Applying Pre-trained Models
Coming soon...
## Instructions on Reproducing Quantitative Results in The Paper
Coming soon...
