# TransMorph on IXI dataset
This is a short description of applying TransMorph varaints and the baseline models on [IXI dataset](https://brain-development.org/ixi-dataset/) for **atlas-to-patient registration**.

All training and inference scripts mentioned in this page are in [`IXI/ (Currently Not Available)`]().

## IXI Dataset:
:exclamation: We will release the pre-processed dataset once we finish all the experiments.\
:exclamation: 12/10/2021 - We are currently testing TransMorph and the baseline models on IXI dataset for atlas-to-patient registration. Once we finish testing, we will make our preprocessed IXI dataset publicly available under the Creative Commons [CC BY-SA 3.0 license](https://creativecommons.org/licenses/by-sa/3.0/legalcode). If you use the data, you should acknowledge the TransMorph paper:

    @article{chen2021transmorph,
    title={TransMorph: Transformer for unsupervised medical image registration},
    author={Chen, Junyu and Du, Yong and He, Yufan and Segars, William P and Li, Ye and Frey, Eric C},
    journal={arXiv preprint arXiv:2111.10480},
    year={2021}
    }

and acknowledge the source of the IXI data: https://brain-development.org/ixi-dataset/ 

- Click on the link to start downloading the pre-processed dataset: [preprocessed IXI dataset (Currently Not Available)]()

- There are **576** brain MRI volumes in total. We split the dataset into a ratio of 7:1:2, where 403 for training (`IXI_data/Train/`), 58 for validation (`IXI_data/Val/`), and 115 for testing (`IXI_data/Test/`).
- There is an atlas image volume and the corresponding subcortical segmentation (`IXI_data/altas.pkl`). This atlas volume was obtained from [CycleMorph](https://github.com/boahK/MEDIA_CycleMorph).
- Each `.pkl` file contains a T1 weighted brain MRI and its corresponding subcortical segmentation. You can read `.pkl` file in python by doing:
```python
import pickle
def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

image, label = pkload("image.pkl")
```

## Pre-trained Weights
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

