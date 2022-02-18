# TransMorph on OASIS dataset @ [MICCAI 2021 Learn2Reg Challenge](https://learn2reg.grand-challenge.org/evaluation/task-3-validation/leaderboard/)

TransMorph ranked ***1st*** on the validation set of task03 (***Validation dataset was NOT used during training***)
<img src="https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/example_imgs/L2R_task03_ep350.jpg" width="600"/>

- This page provides a brief description of applying TransMorph model on the [OASIS dataset](https://github.com/adalca/medical-datasets/blob/master/neurite-oasis.md) ([MICCAI 2021 Learn2Reg Challenge task03](https://learn2reg.grand-challenge.org/evaluation/task-3-validation/leaderboard/)).
- All training and inference scripts mentioned on this page are in [`OASIS/`](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/tree/main/OASIS).
- The OASIS dataset in `.pkl` format is provided in this page.

## [OASIS dataset](https://github.com/adalca/medical-datasets/blob/master/neurite-oasis.md)
This dataset was made available by [Andrew Hoopes](https://www.nmr.mgh.harvard.edu/user/3935749) and [Adrian V. Dalca](http://www.mit.edu/~adalca/) for the following HyperMorph paper.  
If you use this dataset please cite the following and refer to the [OASIS Data Use Agreement](http://oasis-brains.org/#access).

 - [HyperMorph: Amortized Hyperparameter Learning for Image Registration](https://arxiv.org/abs/2101.01035).  
   Hoopes A, Hoffmann M, Fischl B, Guttag J, Dalca AV.   
   IPMI 2021.

 - Open Access Series of Imaging Studies (OASIS): Cross-Sectional MRI Data in Young, Middle Aged, Nondemented, and Demented Older Adults.  
    Marcus DS, Wang TH, Parker J, Csernansky JG, Morris JC, Buckner RL.  
    Journal of Cognitive Neuroscience, 19, 1498-1507.

We converted this dataset into `.pkl` format so that it matches our custom PyTorch [`Dataset`](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/OASIS/TransMorph/data/datasets.py) function.\
You can download the data here: [<img src="https://github.com/junyuchen245/junyuchen245.github.io/blob/master/images/down_arrow.gif" width="30px">](https://drive.google.com/uc?export=download&id=1b7_nHrzPepzm4Mkm1uWDlTZamEmxs4MB)
[Download Dataset from Google Drive (1.11G)](https://drive.google.com/uc?export=download&id=1b7_nHrzPepzm4Mkm1uWDlTZamEmxs4MB)

## Pre-trained Model Weights
Click on the `Model Weights` to start downloading the pre-trained weights.\
We also provided the Tensorboard training log for each model. To visualize loss and validation curves, run: \
```Tensorboard --logdir=*training log file name*``` in terminal. *Note: This requires Tensorboard installation (`pip install tensorboard`).*
### TransMorph:
1. TransMorph-Large ([Val. Dice: 0.8623, 1st ranking](https://learn2reg.grand-challenge.org/evaluation/task-3-validation/leaderboard/)) ([Model Weights (1.69G)](https://drive.google.com/uc?export=download&id=10CnukM9Li5Bh8X5rP_OvfkDio8Mgxj77) | Tensorboard Training Log (N/A))

## Citation:
If you find this code is useful in your research, please consider to cite:
    
    @article{chen2021transmorph,
    title={TransMorph: Transformer for unsupervised medical image registration},
    author={Chen, Junyu and Frey, Eric C and He, Yufan and Segars, William P and Li, Ye and Du, Yong},
    journal={arXiv preprint arXiv:2111.10480},
    year={2021}
    }
