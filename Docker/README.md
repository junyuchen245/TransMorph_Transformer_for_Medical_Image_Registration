# Brain MRI Image Registration with TransMorph
This repository provides Docker images for different versions of TransMorph, a tool for brain MRI image registration tailored to various applications. You can access the Docker images on our [Docker Hub](https://hub.docker.com/repository/docker/jchen245/transmorph_brain_mri_registration/general).

- ***At the moment, only the non-diffeomorphic variant of TransMorph is available, and the Docker image is CPU-based. GPU-enabled containers and other variants will be added soon, so stay tuned!***

## Registration Pipeline
TransMorph is trained on a dataset of brain MRI images, so to use it on new datasets, some preprocessing is necessary to ensure that the intensity ranges and orientations are consistent with the training dataset.\
***Please note that brain MRI images should be skull-stripped and went through bias correction before use. You can accomplish this with various tools, such as [SynthStrip](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/) and [ITK](https://github.com/InsightSoftwareConsortium/ITK).***

The registration pipeline includes the following steps:
1. **Reorientation**: Align the input images (moving and fixed) with a standard template image.
2. **Intensity Normalization**: Scale the intensity values of the input images to the range `[0, 1]`.
3. **Resampling**: Adjust the voxel sizes of the input images to match the template image's dimensions (`1mm x 1mm x 1mm`).
4. **Affine Pre-alignment**: Perform affine alignment of the input images to the template using [ANTsPy](https://github.com/ANTsX/ANTsPy).
5. **Deformable Registration**: Warp the moving image to the fixed image using [TransMorph](https://www.sciencedirect.com/science/article/pii/S1361841522002432)'s deformable registration.
6. **Post-processing (Optional)**: Resample and reorient the results back to the original spaces of the moving or fixed images, as needed.

## Instructions on Running the Docker Image
To use the TransMorph Docker image, start by pulling it from Docker Hub:
  ```bash
  docker pull jchen245/transmorph_brain_mri_registration:transmorph_brain_mri_t1_v0
  ```
Next, run the Docker container with the following command, or use the provided [`test.sh`](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/Docker/test.sh) script (`bash test.sh`):
  ```bash
   docker run --rm  \
        --ipc=host \
        --memory 256g \
        --mount type=bind,source=[path to dataset.json file],target=/input_dataset.json \
        --mount type=bind,source=[path to config.json file],target=/configs_registration.json \
        --mount type=bind,source=[path to input directory],target=/input \
        --mount type=bind,source=[path to output directory],target=/output \
        jchen245/transmorph_brain_mri_registration:transmorph_brain_mri_t1_v0
   ```

### Dataset JSON file
The program reads data from the specified [path to input directory] based on the dataset.json file, which should follow this format:
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
        }
    ]
  }
  ```
- `"label"`: The label map for the moving image (optional). If provided, the program will save the warped label map. An example `dataset.json` file is available in the [repository](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/Docker/test_dataset.json).
### Configuration JSON file
To customize the registration process, modify the [`configurations.json`](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/Docker/configs_registration.json) file as needed:
  ```json
  {
    "n4_bias_correction_moving": false,
    "n4_bias_correction_fixed": false,
    "affine": true,
    "deformable": true,
    "resample": true,
    "resample_back": true,
    "instance_optimization": false,
    "save_registration_inputs": true,
    "IO_iteration": 20,
    "sim_weight": 1,
    "reg_weight": 1,
    "verbose": true
  }
  ```
- `n4_bias_correction_moving`: Toggle N4 bias field correction for moving image.
- `n4_bias_correction_fixed`: Toggle N4 bias field correction for fixed image.
- `affine`: Toggle affine pre-alignment.
- `deformable`: Toggle deformable registration with TransMorph.
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

## All Possible Outputs
- `affine_fwdtransforms.mat`: Affine transformation matrix used by ANTsPy.
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
