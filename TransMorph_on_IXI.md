# TransMorph on IXI dataset
- This page provides a brief description of applying TransMorph variants and the baseline models on the [IXI dataset](https://brain-development.org/ixi-dataset/) for **atlas-to-patient registration**.
- All training and inference scripts mentioned on this page are in [`IXI/`](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/tree/main/IXI).
- This page contains our preprocessed IXI dataset (including subcortical segmentations).
## IXI Dataset:
<img src="https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/example_imgs/IXI_dataset.jpg" width="1000"/>\
<a rel="license" href="http://creativecommons.org/licenses/by-sa/3.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/3.0/88x31.png" /></a>\
:exclamation: 12/10/2021 - We are currently testing TransMorph and the baseline models on the IXI dataset for atlas-to-patient registration. Once we finish testing, we will make our preprocessed IXI dataset and the pre-trained models publicly available.\
:exclamation: Our preprocessed IXI dataset is made available under the <a rel="license" href="http://creativecommons.org/licenses/by-sa/3.0/">Creative Commons Attribution-ShareAlike 3.0 Unported License</a>. If you use this dataset, you should acknowledge the TransMorph paper:

    @article{chen2021transmorph,
    title={TransMorph: Transformer for unsupervised medical image registration},
    author={Chen, Junyu and Frey, Eric C and He, Yufan and Segars, William P and Li, Ye and Du, Yong},
    journal={arXiv preprint arXiv:2111.10480},
    year={2021}
    }

and acknowledge the source of the IXI data: https://brain-development.org/ixi-dataset/ 

- ***Preprocessing:*** The IXI dataset was preprocessed (e.g., skull stripping, affine alignment, and subcortical segmentation) by using [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/fswiki). The steps we used are listed here - <a href="https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/PreprocessingMRI.md">Brain MRI preprocessing and subcortical segmentation using FreeSurfer</a>
- ***Train-Val-Test split:*** There are **576** brain MRI volumes in total. We split the dataset into a ratio of **7:1:2**, where **403** for training (`IXI_data/Train/`), **58** for validation (`IXI_data/Val/`), and **115** for testing (`IXI_data/Test/`).
- ***Atlas image:*** Additionally, there is one atlas MRI volume and its corresponding subcortical segmentation (`IXI_data/altas.pkl`). This atlas volume was obtained from [CycleMorph](https://github.com/boahK/MEDIA_CycleMorph).
- ***File format:*** Each `.pkl` file contains a T1 weighted brain MRI and its corresponding subcortical segmentation. Learn more about `.pkl` format [here](https://www.datacamp.com/community/tutorials/pickle-python-tutorial). You can read `.pkl` file in python by doing:
    ```python
    import pickle
    def pkload(fname):
        with open(fname, 'rb') as f:
            return pickle.load(f)

    image, label = pkload("subject_0.pkl") 
    # image: a preprocessed T1-weighted brain MRI volume. Shape: 160 x 192 x 224 Intensity: [0,1]
    # label: the corresponding subcortical segmentations. Shape: 160 x 192 x 224 Intensity: Integers
    ```
- ***Label map:*** A description of each label and the corresponding indexing value is provided [here](https://github.com/junyuchen245/Preprocessed_IXI_Dataset/blob/main/label_info.txt).
- ***Image size:*** Each image and label map has a size of `160 x 192 x 224`.
- ***Normalization:*** The intensity values of each image volume are normalized into a range `[0,1]`.
- ***Dataset structure:***
    ```bash
    IXI/Train/------
            subject_0.pkl   <--- a T1 brain MR image and its label map
            subject_4.pkl
            .......
    IXI/Val/------
            subject_2.pkl
            subject_5.pkl
            .......
    IXI/Test/------
            subject_1.pkl
            subject_3.pkl
            .......
    IXI/Test/atlas.pkl      <--- Atlas image and its label map
    ```
## Download Dataset: [<img src="https://github.com/junyuchen245/junyuchen245.github.io/blob/master/images/down_arrow.gif" width="30px">]()
[Download Dataset from Google Drive (Currently Not Available)]()

## Pre-trained Model Weights
Click on the `Model Weights` to start downloading the pre-trained weights.\
We also provided the Tensorboard training log for each model. To visualize loss and validation curves, run: \
```Tensorboard --logdir=*training log file name*``` in terminal. *Note: This requires Tensorboard installation (`pip install tensorboard`).*
### TransMorph Variants:
1. TransMorph ([Model Weights (0.8G)](https://drive.google.com/uc?export=download&id=1SDWj2ppvmkXMn1qw8jFkAeQqW3B8VZcu) | [Tensorboard Training Log (1.7G)](https://drive.google.com/uc?export=download&id=1tFCODnHGY08mEON2Oy9P54tK0coyIrw8))
2. TransMorph-Bayes ([Model Weights (0.9G)](https://drive.google.com/uc?export=download&id=1TxCFeUokywV5kff_A1EjrCY6QjH_jFgb) | [Tensorboard Training Log (1.9G)](https://drive.google.com/uc?export=download&id=1G3XOSBgyjdBWp_Dbz8urKtn-zoKwZtd8))
3. TransMorph-diff ([Model Weights (0.5G)](https://drive.google.com/uc?export=download&id=1K_6-CS_x7tkgYQWXGMhGIhksk83pCBu4) | [Tensorboard Training Log (1.9G)](https://drive.google.com/uc?export=download&id=1TZU6pIDn3KLZzoNitcOTs-O6dOEKWJWu))
4. TransMorph-bspl ([Model Weights (0.7G)](https://drive.google.com/uc?export=download&id=1ZLNM9KUP8KtCXBLwXRc9dx3OdWft6eMY) | [Tensorboard Training Log (1.6G)](https://drive.google.com/uc?export=download&id=1ZJvyVRghLsEapMJZQlR-osvriywk56ed))

### Baseline Models:
***Pre-trained baseline registration models:***
1. VoxelMorph-1 ([Model Weights (83M)](https://drive.google.com/uc?export=download&id=1pjujL0PTELYy3TS_nj0BFnJjBF7OUqqm) | [Tensorboard Training Log (1.6G)](https://drive.google.com/uc?export=download&id=1Io7MvpaUlMfH1A2ZuWX4Mbc0uJaAhl-Q))
2. VoxelMorph-2 ([Model Weights (83.4M)](https://drive.google.com/uc?export=download&id=1awrgjTWCykjpMlBVUbvILBVpZTzBXd4V) | [Tensorboard Training Log (1.6G)](https://drive.google.com/uc?export=download&id=1-yU4-XMbStHW1FGWkiOYIc0kNEToByy0))
3. VoxelMorph-diff ([Model Weights (3.5M)](https://drive.google.com/uc?export=download&id=1Dv6Z1MK_JU6dveGHu6jkY3VRUuiXRFG8) | [Tensorboard Training Log (1.8G)](https://drive.google.com/uc?export=download&id=1n6RnPW9WQzA-uzKq3HGZoUHVterJMccS))
4. CycleMorph ([Model Weights (1.4M)](https://drive.google.com/uc?export=download&id=1Fzs9pGKmlYtozCNfKTYnN3_yx34t61x8) | [Tensorboard Training Log (1.7G)](https://drive.google.com/uc?export=download&id=1N44XcmftAg62uFmb7LamJvmphu2DKMql))
5. MIDIR ([Model Weights (4.1M)](https://drive.google.com/uc?export=download&id=1JWCF1pqmF2FE8mc0MVP31y3KKQ08M-fM) | [Tensorboard Training Log (1.6G)](https://drive.google.com/uc?export=download&id=1nFq8XchhqJPipT1fIuE9pkUYSMSlozzU))

***Pre-trained baseline Transformer-based registration models:***
1. PVT ([Model Weights (1.0G)](https://drive.google.com/uc?export=download&id=1AlfgG9zStsz5n4mzJvGDtyQaKtHkmp6b) | [Tensorboard Training Log (1.7G)](https://drive.google.com/uc?export=download&id=1KvkehhNnDpS2gsRv5fdiyaHXyzWsALV5))
2. nnFormer ([Model Weights (0.7G)](https://drive.google.com/uc?export=download&id=1doB6-qia1WnPy2UV1g95LDM7GjYsJV4n) | [Tensorboard Training Log (1.7G)](https://drive.google.com/uc?export=download&id=1nC2eKgCD5aX2G7rZJdw3UukOsd6dR29I))
3. CoTr ([Model Weights (670M)](https://drive.google.com/uc?export=download&id=1aSZ9oN6Fn7iZYlk-o0UBJ_Mf70cUHgP3) | [Tensorboard Training Log (1.7G)](https://drive.google.com/uc?export=download&id=1pZ7qPjQBxjMTQJemIzap-bBJE170ZSdC))
4. ViT-V-Net ([Model Weights (561M)](https://drive.google.com/uc?export=download&id=1VBL2nlB3iWcDBQbOr4FxhR-xrG_quIbT) | [Tensorboard Training Log (1.7G)](https://drive.google.com/uc?export=download&id=1cfuTBUaJYhpcalU5-aMULl3JjCaV0H3H))

***Validation Dice Scores During Training***

<img src="https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/example_imgs/validation_dice_IXI.jpg" width="600"/>

## Instructions on Applying Pre-trained Models
### Step 1
Create the directories shown below. After that, put the pretrained models in the corresponding directories:
```bash
IXI/TransMorph/------
            experiments/TransMorph_ncc_1_diffusion_1/
            experiments/TransMorphBayes_ncc_1_diffusion_1/
            experiments/TransMorphDiff/
            experiments/TransMorphBSpline_ncc_1_diffusion_1/

IXI/Baseline_Transformers/------
            experiments/CoTr_ncc_1_diffusion_1/
            experiments/PVT_ncc_1_diffusion_1/
            experiments/ViTVNet_ncc_1_diffusion_1/
            experiments/nnFormer_ncc_1_diffusion_1/

IXI/Baseline_registration_methods/------
            CycleMorph/experiments/CycleMorph/
            MIDIR/experiments/MIDIR_ncc_1_diffusion_1/
            VoxelMorph/experiments/------
                        Vxm_1_ncc_1_diffusion_1/
                        Vxm_2_ncc_1_diffusion_1/
            VoxelMorph-diff/experiments/VxmDiff/
```
### Step 2
Change the directories in the inference scripts (`infer_xxx.py`) to the IXI dataset folder:
```python
atlas_dir = 'Path_to_IXI_data/atlas.pkl'
test_dir = 'Path_to_IXI_data/Val/'
```
The inference scripts are located in:
```bash
IXI/TransMorph/------
            infer_TransMorph.py
            infer_TransMorph_Bayes.py
            infer_TransMorph_bspl.py
            infer_TransMorph_diff.py

IXI/Baseline_Transformers/------
            infer_CoTr.py
            infer_nnFormer.py
            infer_PVT.py
            infer_nnFormer.py

IXI/Baseline_registration_methods/------
            CycleMorph/infer.py
            MIDIR/infer.py
            VoxelMorph/infer.py
            VoxelMorph-diff/infer.py
```
### Step 3
At the bottom of the inference scripts, specify the GPU to be used for evaluation:
```python
'''
GPU configuration
'''
GPU_iden = 0
GPU_num = torch.cuda.device_count()
```
### Step 4
Make sure that the folder containing the pre-trained model corresponds to the one used in the inference scripts. For example, for [TransMorph](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/78d90ec01f463c4f07607b3567c844eed9d04c07/IXI/TransMorph/infer_TransMorph.py#L17-L19):
```python
weights = [1, 1]
model_folder = 'TransMorph_ncc_{}_diffusion_{}/'.format(weights[0], weights[1])
model_dir = 'experiments/' + model_folder
```
In terminal, run: `python -u IXI/Path_to_Model/infer_xxx.py`. The results (a `.csv` file with the name of the model that contains Dice scores) will be saved in a sub-folder called `IXI/Path_to_Model/Quantitative_Results/`.
## Instructions on Reproducing Quantitative Results in the Paper
- Once the evaluation scripts have been run, copy the result '.csv' files to 'IXI/Results/' for producing mean and std. Dice scores, percentage of non-pos. Jecobian determinants, and boxplots for subcortical segmentations.
- Our results (i.e., the `.csv` files) are provided in `IXI/Results/`. To visualize boxplots, simply run `python -u IXI/analysis.py` and `python -u IXI/analysis_trans.py`.
- To plot your own results, simply replace the files in `IXI/Results/`. If the file names are different, you will need to modify the names used in [`IXI/analysis.py`](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/12ea90a8129fc167af22eafb5c0272bdc4141b41/IXI/analysis.py#L17) and [`IXI/analysis_trans.py`](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/12ea90a8129fc167af22eafb5c0272bdc4141b41/IXI/analysis_trans.py#L17).
## Quantitative Results
|Model |Dice|% of \|J\|<=0|
|---|---|---|
|Affine|0.386±0.195|-|
|SyN|0.639±0.151|<0.0001|
|NiftyReg|0.640±0.166|<0.0001|
|LDDMM|0.675±0.135|<0.0001|
|deedsBCV|0.733±0.126|0.147±0.050|
|VoxelMorph-1|0.723±0.130|1.590±0.339|
|VoxelMorph-2|0.726±0.123|1.522±0.336|
|VoxelMorph-diff|0.577±0.165|<0.0001|
|CycleMorph|0.730±0.124|1.719±0.382|
|MIDIR|0.736±0.129|<0.0001|
|ViT-V-Net|0.728±0.124|1.609±0.319|
|CoTr|0.721±0.128|1.858±0.314|
|PVT|0.729±0.135|1.292±0.342|
|nnFormer|0.740±0.134|1.595±0.358|
|TransMorph|0.746±0.128|1.579±0.328|
|TransMorph-Bayes|0.746±0.123|1.560±0.333|
|TransMorph-bspl|0.752±0.128|<0.0001|
|TransMorph-diff|0.599±0.156|<0.0001|

<img src="https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/example_imgs/IXI_dsc_conventional.jpg" width="900"/>
<img src="https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/example_imgs/IXI_dsc_transformer.jpg" width="900"/>

## Qualitative Results
<img src="https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/example_imgs/IXI_Brain.jpg" width="900"/>
