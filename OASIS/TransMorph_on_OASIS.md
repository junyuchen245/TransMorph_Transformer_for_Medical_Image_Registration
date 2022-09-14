# TransMorph on OASIS dataset @ [MICCAI 2021 Learn2Reg Challenge](https://learn2reg.grand-challenge.org/evaluation/task-3-validation/leaderboard/)

TransMorph ranked ***1st*** on the validation & **test** set of task03 (***Validation dataset was NOT used during training***)
<img src="https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/example_imgs/L2R_task03_TransMorphLargeCas.jpg" width="600"/>

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
You can download the data here: [<img src="https://github.com/junyuchen245/junyuchen245.github.io/blob/master/images/down_arrow.gif" width="30">](https://drive.google.com/uc?export=download&id=1BdEaylMDpeXtyuX5QH8l_Ut4OgenKss4)
[Download Dataset from Google Drive (1.3G)](https://drive.google.com/uc?export=download&id=1BdEaylMDpeXtyuX5QH8l_Ut4OgenKss4)

## Pre-trained Model Weights
Click on the `Model Weights` to start downloading the pre-trained weights.\
We also provided the Tensorboard training log for each model. To visualize loss and validation curves, run: \
```Tensorboard --logdir=*training log file name*``` in terminal. *Note: This requires Tensorboard installation (`pip install tensorboard`).*

### Validation set results
|Ranking|Model|Dice|SDlogJ|HdDist95|Pretrained Weights|Tensorboard Log|
|---|---|---|---|---|---|---|
|[1](https://learn2reg.grand-challenge.org/evaluation/task-3-validation/leaderboard/)|[TM-TVF](https://github.com/junyuchen245/TransMorph_TVF)|0.8691 ± 0.0145|0.0945|1.3969|N/A| N/A|
|[2](https://learn2reg.grand-challenge.org/evaluation/task-3-validation/leaderboard/)|TransMorph-Large|0.8623 ± 0.0144|0.1276|1.4315|[Model Weights (1.69G)](https://drive.google.com/uc?export=download&id=10CnukM9Li5Bh8X5rP_OvfkDio8Mgxj77)| [Tensorboard Training Log (1.52G)](https://drive.google.com/uc?export=download&id=197glDrMjiyhE1AMV3-YKzwlO2CS1UmCC)|
|3|TransMorph|0.8575 ± 0.0145|0.1253|1.4594|[Model Weights (0.8G)](https://drive.google.com/uc?export=download&id=1FRDmfDreHBsvuYdCeUwauU80CWGsiUb7)| [Tensorboard Training Log (1.52G)](https://drive.google.com/uc?export=download&id=16CF85KNIXOHc27uG4l34dWjfWW8aost-)|

*: Scripts and pretrained model for TransMorph-Large-Cascade will be made available after publication.

### Test set results (*results obtained from Learn2Reg challenge organizers*)
|Ranking|Model|Dice|SDlogJ|HdDist95|
|---|---|---|---|---|
|1|[TM-TVF](https://github.com/junyuchen245/TransMorph_TVF)|**0.8241 ± 0.1516**|0.0905 ± 0.0054|**1.6329 ± 0.4358**|
|2|TransMorph-Large|*0.8196 ± 0.1497*|0.1244 ± 0.0148|1.6564 ± 1.7368|
|3|TransMorph|0.8162 ± 0.1541| 0.1242 ± 0.0136|1.6920 ± 1.7587|
|4|[LapIRN](https://github.com/cwmok/LapIRN)|0.82| 0.07 |1.67|
|5|[ConvexAdam](https://github.com/multimodallearning/convexAdam)|0.81| 0.07 |1.63|
...


## Instructions on Applying Pre-trained Models
### Step 1
Create the directories shown below. After that, put the pretrained models in the corresponding directories:
```bash
OASIS/TransMorph/------
            experiments/TransMorph_ncc_1_dsc_1_diffusion_1/
            experiments/TransMorphLarge_ncc_1_dsc_1_diffusion_1/
```
### Step 2
Change the directories of the validation/test data and the output deformations the [`submit_TransMorph.py`](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/OASIS/TransMorph/submit_TransMorph.py):
https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/d5e842ee87def4cef0425fd090ebbc959918abe6/OASIS/TransMorph/submit_TransMorph.py#L15-L16

### Step 3
Change the folder name and model name in the [`submit_TransMorph.py`](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/OASIS/TransMorph/submit_TransMorph.py) :
https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/d5e842ee87def4cef0425fd090ebbc959918abe6/OASIS/TransMorph/submit_TransMorph.py#L19-L22

### Step 4
Zip the output folder and submit the zip file to [Learn2Reg challenge Task03](https://learn2reg.grand-challenge.org/evaluation/task-3-validation/submissions/create/).

## Citation
If you find this code is useful in your research, please consider to cite:
    
    @article{chen2022transmorph,
    title = {TransMorph: Transformer for unsupervised medical image registration},
    journal = {Medical Image Analysis},
    pages = {102615},
    year = {2022},
    issn = {1361-8415},
    doi = {https://doi.org/10.1016/j.media.2022.102615},
    url = {https://www.sciencedirect.com/science/article/pii/S1361841522002432},
    author = {Junyu Chen and Eric C. Frey and Yufan He and William P. Segars and Ye Li and Yong Du}
    }
