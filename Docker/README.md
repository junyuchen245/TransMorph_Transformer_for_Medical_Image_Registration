# Brain MRI Image Registration with TransMorph
This repository provides Docker images for different versions of TransMorph, a tool for brain MRI image registration tailored to various applications. You can access the Docker images on our [Docker Hub](https://hub.docker.com/repository/docker/jchen245/transmorph_brain_mri_registration/general).

## Registration Pipeline
TransMorph is trained on a dataset of brain MRI images, so to use it on new datasets, some preprocessing is necessary to ensure that the intensity ranges and orientations are consistent with the training dataset. The registration pipeline includes the following steps:
1. **Reorientation**: Align the input images (moving and fixed) with a standard template image.
2. **Intensity Normalization**: Scale the intensity values of the input images to the range `[0, 1]`.
3. **Resampling**: Adjust the voxel sizes of the input images to match the template image's dimensions (`1mm x 1mm x 1mm`).
4. **Affine Pre-alignment**: Perform affine alignment of the input images to the template using [ANTsPy](https://github.com/ANTsX/ANTsPy).
5. **Deformable Registration**: Warp the moving image to the fixed image using [TransMorph](https://www.sciencedirect.com/science/article/pii/S1361841522002432)'s deformable registration.
6. **Post-processing (Optional)**: Resample and reorient the results back to the original spaces of the moving or fixed images, as needed.

## Instructions on Running the Docker Image
You can run Docker image using the following command or alternatively, you can use the provided bash script to run it (i.e., `bash test.sh`)
  ```bash
   docker run --rm  \
        --ipc=host \
        --memory 256g \
        --mount type=bind,source=/scratch/jchen/python_projects/TransMorph_brain_registration/test_dataset.json,target=/input_dataset.json \
        --mount type=bind,source=/scratch/jchen/python_projects/TransMorph_brain_registration/configs_registration.json,target=/configs_registration.json \
        --mount type=bind,source=/scratch/jchen/python_projects/TransMorph_brain_registration/test_data,target=/input \
        --mount type=bind,source=/scratch/jchen/python_projects/TransMorph_brain_registration/test_output,target=/output \
        transmorph_brain_mri_t1
   ```

