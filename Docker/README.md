# Brain MRI Image Registration with TransMorph
This repository provides Docker images for different variants of TransMorph, a tool for brain MRI image registration for various applications. You can access the Docker images on our [Docker Hub](https://hub.docker.com/repository/docker/jchen245/transmorph_brain_mri_registration/general).

- ***This TransMorph Docker image currently only supports mono-modality registration.***

## Registration Pipeline
TransMorph is trained on a dataset of brain MRI images, so to use it on new datasets, some preprocessing is necessary to ensure that the intensity ranges and orientations are consistent with the training dataset.\
***Please note that brain MRI images should be skull-stripped. You can accomplish this with various tools, such as [SynthStrip](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/).***

The registration pipeline includes the following steps:
1. **Reorientation**: Align the input images (moving and fixed) with a standard template image.
2. **N4 Bias Field Correction**: Apply N4 bias field correction to the input images using [ANTsPy](https://github.com/ANTsX/ANTsPy).
3. **Intensity Normalization**: Scale the intensity values of the input images to the range `[0, 1]` using [intensity-normalization](https://github.com/jcreinhold/intensity-normalization/tree/master) tool.
4. **Resampling**: Adjust the voxel sizes of the input images to match the template image's dimensions (`1mm x 1mm x 1mm`).
5. **Affine Pre-alignment**: Perform affine alignment of the input images to the template using [ANTsPy](https://github.com/ANTsX/ANTsPy).
6. **Deformable Registration**: Warp the moving image to the fixed image using [TransMorph](https://www.sciencedirect.com/science/article/pii/S1361841522002432)'s deformable registration.
7. **Post-processing (Optional)**: Resample and reorient the results back to the original spaces of the moving or fixed images, as needed.

## Instructions on Running the Docker Image
To use the CPU-based TransMorph Docker image, start by pulling it from Docker Hub:
  ```bash
  docker pull jchen245/transmorph_brain_mri_registration:transmorph_brain_mri_t1_v2
  ```
For the GPU-based Docker image, use:
```bash
  docker pull jchen245/transmorph_brain_mri_registration:transmorph_brain_mri_t1_v2_gpu
  ```
Next, run the CPU-based Docker image with the following command, or use the provided [`test.sh`](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/Docker/test.sh) script (`bash test.sh`):
  ```bash
   docker run --rm  \
        --ipc=host \
        --memory 256g \
        --mount type=bind,source=[path to dataset.json file],target=/input_dataset.json \
        --mount type=bind,source=[path to config.json file],target=/configs_registration.json \
        --mount type=bind,source=[path to input directory],target=/input \
        --mount type=bind,source=[path to output directory],target=/output \
        jchen245/transmorph_brain_mri_registration:transmorph_brain_mri_t1_v2
   ```
For the GPU-based version, use this command or run the provided [`test_GPU.sh`](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/Docker/test.sh) script (`bash test_GPU.sh`):
  ```bash
   docker run --rm \
        --ipc=host \
        --memory 256g \
        --gpus "device=0"\
        --mount type=bind,source=[path to dataset.json file],target=/app/input_dataset.json \
        --mount type=bind,source=[path to config.json file],target=/app/configs_registration.json \
        --mount type=bind,source=[path to input directory],target=/app/input \
        --mount type=bind,source=[path to output directory],target=/app/output \
        transmorph_brain_mri_t1_gpu
   ```

### Dataset JSON file
The program reads data from the specified `[path to input directory]` based on the `dataset.json` file, which should follow this format:
  ```json
  {
    "inputs": [
        {
            "fixed": "./Example_0_img.nii.gz",
            "moving": "./Example_1_img.nii.gz"
        },
        {
            "fixed": "./Example_1_img.nii.gz",
            "moving": "./Example_2_img.nii.gz",
            "label": "./Example_2_lbl.nii.gz"
        },
        {
            "fixed": "./Example_2_img.nii.gz",
            "moving": "./Example_3_img.nii.gz"
        },
        {
            "fixed": "./Example_3_img.nii.gz",
            "moving": "./Example_4_img.nii.gz",
            "label": "./Example_4_lbl.nii.gz"
        },
        {
            "fixed": "./IXI107-Guys_T2.nii.gz",
            "moving": "./IXI128-HH_T2.nii.gz",
            "label": "./IXI128-HH_label.nii.gz",
            "fixed_modality": "T2",
            "moving_modality": "T2",
            "fixed_brain_mask": "./IXI107-Guys_bmask.nii.gz",
            "moving_brain_mask": "./IXI128-HH_bmask.nii.gz",
            "fixed_scaling_factor": 1,
            "moving_scaling_factor": 1
        }
    ]
  }
  ```
- `"label"` (optional): The label map for the moving image. If provided, the program will save the warped label map. An example `dataset.json` file is available in the [repository](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/Docker/test_dataset.json).
- `"fixed_modality"` (optional): The imaging modality of the fixed image. Defaults to "T1" if not specified.
- `"moving_modality"` (optional): The imaging modality of the moving image. Defaults to "T1" if not specified.
- `"fixed_brain_mask"` (optional): Brain mask for skull-stripping the fixed image. If not provided, no mask will be used.
- `"moving_brain_mask"` (optional): Brain mask for skull-stripping the moving image. If not provided, no mask will be used.
- `"fixed_scaling_factor"` (optional): Scaling factor for adjusting the intensity range of the fixed image (i.e., `img=img/fixed_scaling_factor`). The default is 255, assuming white matter is normalized to 110 (i.e., `img=img/255.`).
- `"moving_scaling_factor"` (optional): Scaling factor for adjusting the intensity range of the moving image (i.e., `img=img/fixed_scaling_factor`). Defaults to 255 if not set.

**Please note that the current TransMorph Docker image is trained on T1-weighted brain MRI scans. While it has been tested on other modalities such as T2 or PD, optimal performance is not guaranteed. For those cases, instance optimization is recommended.**
### Configuration JSON file
To customize the registration process, modify the [`configurations.json`](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/Docker/configs_registration.json) file as needed:
  ```json
  {
    "n4_bias_correction_moving": false,
    "n4_bias_correction_fixed": false,
    "intensity_normalization": true,
    "affine": true,
    "affine_type": "Affine",
    "affine_metric": "mattes",
    "affine_iteration": [2100, 1200, 1200, 10],
    "affine_shrink_factor": [6, 4, 2, 1],
    "affine_smoothing_sigmas":[3, 2, 1, 0],
    "deformable": true,
    "diffeomorphic": true,
    "resample": true,
    "resample_back": true,
    "instance_optimization": true,
    "save_registration_inputs": true,
    "IO_iteration": 20,
    "sim_weight": 1,
    "reg_weight": 1,
    "verbose": true
  }
  ```
- `n4_bias_correction_moving`: Toggle N4 bias field correction for moving image.
- `n4_bias_correction_fixed`: Toggle N4 bias field correction for fixed image.
- `intensity_normalization`: Toggle intensity normalization for both images.
- `affine`: Toggle affine pre-alignment.
- `affine_metric`: Specifies the image similarity metric used for affine registration with the ANTs package (default: "mattes").
- `affine_iteration`: Defines the number of iterations for affine registration at different scales (default: [2100, 1200, 1200, 10]).
- `affine_shrink_factor`: Sets the shrink factor for affine registration across different scales (default: [6, 4, 2, 1]).
- `affine_smoothing_sigmas`: Determines the smoothing factor for affine registration at various scales (default: [3, 2, 1, 0]).
- `deformable`: Toggle deformable registration with TransMorph.
- `diffeomorphic`: Toggle diffeomorphic registration. Note that non-diffeomorphic variant may provide better alignment in terms of Dice and TRE.
- `resample`: Enable resampling to match TransMorph conventions (recommended).
- `resample_back`: Apply resampling to return the results to the original image space.
- `instance_optimization`: Enable optimization for each image pair. This provides better results but increases runtime.
- `save_registration_inputs`: Save intermediate images (e.g., resampled or realigned images) for debugging.
- `IO_iteration`: Number of iterations for instance optimization (default: 20).
- `sim_weight`: Weight for similarity measure in instance optimization (default: 1).
- `reg_weight`: Weight for deformation regularity in instance optimization (default: 1).
- `verbose`: Toggle detailed logging output.

## Example Data for Testing
Example data is available for you to test the Docker image using the provided JSON files. You can access it [here](https://drive.google.com/uc?export=download&id=1hjpXnEFHfyI5nMJie7p0J9f-BYlPQN2c).

## All Possible Output Files
- `affine_fwdtransforms.mat`: Affine transformation matrix used by [ANTsPy](https://github.com/ANTsX/ANTsPy).
- `deformed_moving_image_original_fixed_space.nii.gz`: Deformed moving image transformed back to the original space of the fixed image.
- `deformed_moving_image_original_moving_space.nii.gz`: Deformed moving image transformed back to the original space of the moving image.
- `deformed_moving_image.nii.gz`: Deformed moving image in the template space.
- `deformed_moving_label_original_fixed_space.nii.gz`: Deformed label map of the moving image in the original space of the fixed image.
- `deformed_moving_label_original_moving_space.nii.gz`: Deformed label map of the moving image in the original space of the moving image.
- `deformed_moving_label.nii.gz`: Deformed label map of the moving image in the template space.
- `displacement_field.nii.gz`: Displacement field generated by TransMorph.
- `fixed_image_final.nii.gz`: Fixed image after preprocessing, ready for input to TransMorph.
- `fixed_image_reoriented.nii.gz`: Fixed image reoriented to match the template image’s orientation.
- `moving_image_final.nii.gz`: Moving image after preprocessing, ready for input to TransMorph.
- `moving_image_reoriented.nii.gz`: Moving image reoriented to match the template image’s orientation.
- `moving_label_final.nii.gz`: Label map of the moving image after preprocessing, ready for input to TransMorph.
- `moving_label_reoriented.nii.gz`: Label map of the moving image reoriented to match the template image’s orientation.
