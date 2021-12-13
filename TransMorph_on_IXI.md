# TransMorph on IXI dataset
- This page provides a brief description of applying TransMorph variants and the baseline models on the [IXI dataset](https://brain-development.org/ixi-dataset/) for **atlas-to-patient registration**.\
- All training and inference scripts mentioned on this page are in [`IXI/ (Currently Not Available)`]().

## IXI Dataset:
<img src="https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/example_imgs/IXI_dataset.jpg" width="1000"/>\
:exclamation: 12/10/2021 - We are currently testing TransMorph and the baseline models on the IXI dataset for atlas-to-patient registration. Once we finish testing, we will make our preprocessed IXI dataset publicly available\
:exclamation: Our preprocessed IXI dataset is made available under the Creative Commons [CC BY-SA 3.0 license](https://creativecommons.org/licenses/by-sa/3.0/legalcode). If you use this dataset, you should acknowledge the TransMorph paper:

    @article{chen2021transmorph,
    title={TransMorph: Transformer for unsupervised medical image registration},
    author={Chen, Junyu and Du, Yong and He, Yufan and Segars, William P and Li, Ye and Frey, Eric C},
    journal={arXiv preprint arXiv:2111.10480},
    year={2021}
    }

and acknowledge the source of the IXI data: https://brain-development.org/ixi-dataset/ 

- ***Preprocessing:*** The IXI dataset was preprocessed (e.g., skull stripping, affine alignment, and subcortical segmentation) by using [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/fswiki). The preprocessing steps we used in FreeSurfer are listed here - <a href="https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/PreprocessingMRI.md">Brain MRI preprocessing and subcortical segmentation using FreeSurfer</a>
- ***Train-Val-Test split:*** There are **576** brain MRI volumes in total. We split the dataset into a ratio of **7:1:2**, where **403** for training (`IXI_data/Train/`), **58** for validation (`IXI_data/Val/`), and **115** for testing (`IXI_data/Test/`).
- ***Atlas image:*** Additionally, there is one atlas MRI volume and its corresponding subcortical segmentation (`IXI_data/altas.pkl`). This atlas volume was obtained from [CycleMorph](https://github.com/boahK/MEDIA_CycleMorph).
- ***File format:*** Each `.pkl` file contains a T1 weighted brain MRI and its corresponding subcortical segmentation. You can read `.pkl` file in python by doing:
```python
import pickle
def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

image, label = pkload("image.pkl")
```
- ***Image size:*** Each image and label map has a size of `160 x 192 x 224`.
- ***Download <img src="https://github.com/junyuchen245/junyuchen245.github.io/blob/master/images/down_arrow.gif" width="30px">:*** [Download Dataset (Currently Not Available)]()

## Pre-trained Model Weights
Click on the `Model Weights` to start downloading the pre-trained weights.\
We also provided the Tensorboard training log for each model. To visualize loss and validation curves, run: \
```Tensorboard --logdir=*training log file name*``` in terminal. *Note: Requires Tensorboard installation (`pip install tensorboard`).*
### TransMorph Variants:
1. TransMorph ([Model Weights (0.8G)](https://drive.google.com/uc?export=download&id=1SDWj2ppvmkXMn1qw8jFkAeQqW3B8VZcu) | [Tensorboard Training Log (1.7G)](https://drive.google.com/uc?export=download&id=1tFCODnHGY08mEON2Oy9P54tK0coyIrw8))
2. TransMorph-Bayes ([Model Weights (0.9G)](https://drive.google.com/uc?export=download&id=1TxCFeUokywV5kff_A1EjrCY6QjH_jFgb) | [Tensorboard Training Log (1.9G)](https://drive.google.com/uc?export=download&id=1G3XOSBgyjdBWp_Dbz8urKtn-zoKwZtd8))
3. TransMorph-diff ([Model Weights (0.5G)](https://drive.google.com/uc?export=download&id=1K_6-CS_x7tkgYQWXGMhGIhksk83pCBu4) | [Tensorboard Training Log (1.9G)](https://drive.google.com/file/d/1TZU6pIDn3KLZzoNitcOTs-O6dOEKWJWu/view?usp=sharing))
4. TransMorph-bspl ([Model Weights (0.7G)](https://drive.google.com/uc?export=download&id=1ZLNM9KUP8KtCXBLwXRc9dx3OdWft6eMY) | [Tensorboard Training Log (1.6G)](https://drive.google.com/uc?export=download&id=1ZJvyVRghLsEapMJZQlR-osvriywk56ed))

### Baseline Models:
***Pre-trained baseline registration models:***
1. VoxelMorph-1 ([Model Weights (83M)](https://drive.google.com/uc?export=download&id=1pjujL0PTELYy3TS_nj0BFnJjBF7OUqqm) | [Tensorboard Training Log (1.6G)](https://drive.google.com/uc?export=download&id=1Io7MvpaUlMfH1A2ZuWX4Mbc0uJaAhl-Q))
2. VoxelMorph-2 ([Model Weights (83.4M)](https://drive.google.com/uc?export=download&id=1awrgjTWCykjpMlBVUbvILBVpZTzBXd4V) | [Tensorboard Training Log (1.6G)](https://drive.google.com/uc?export=download&id=1-yU4-XMbStHW1FGWkiOYIc0kNEToByy0))
3. VoxelMorph-diff ([Model Weights (3.5M)](https://drive.google.com/uc?export=download&id=1Dv6Z1MK_JU6dveGHu6jkY3VRUuiXRFG8) | [Tensorboard Training Log (1.8G)](https://drive.google.com/uc?export=download&id=1n6RnPW9WQzA-uzKq3HGZoUHVterJMccS))
4. CycleMorph (Under testing)
5. MIDIR ([Model Weights (4.1M)](https://drive.google.com/uc?export=download&id=1JWCF1pqmF2FE8mc0MVP31y3KKQ08M-fM) | [Tensorboard Training Log (1.6G)](https://drive.google.com/uc?export=download&id=1nFq8XchhqJPipT1fIuE9pkUYSMSlozzU))

***Pre-trained baseline Transformer-based registration models:***
1. PVT (Under testing)
2. nnFormer (Under testing)
3. CoTr (Under testing)
4. ViT-V-Net (Under testing)

## Instructions on Applying Pre-trained Models
Coming soon...
## Instructions on Reproducing Quantitative Results in The Paper
Coming soon...
## Quantitative Results
Coming soon...
|Model |Dice|% of \|J\|<=0|
|---|---|---|
|Affine|||
|SyN|||
|NiftyReg|||
|LDDMM|||
|deedsBCV|||
|VoxelMorph-1|||
|VoxelMorph-2|||
|VoxelMorph-diff|||
|CycleMorph|||
|MIDIR|||
|ViT-V-Net|||
|CoTr|||
|PVT|||
|nnFormer|||
|TransMorph|||
|TransMorph-Bayes|||
|TransMorph-bspl|||
|TransMorph-diff|||
