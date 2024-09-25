#!/usr/bin/env bash
docker pull jchen245/transmorph_brain_mri_registration:transmorph_brain_mri_t1_v2

docker run --rm  \
        --ipc=host \
        --memory 256g \
        --mount type=bind,source=/scratch/jchen/python_projects/TransMorph_brain_registration/test_dataset.json,target=/input_dataset.json \
        --mount type=bind,source=/scratch/jchen/python_projects/TransMorph_brain_registration/configs_registration.json,target=/configs_registration.json \
        --mount type=bind,source=/scratch/jchen/python_projects/TransMorph_brain_registration/test_data,target=/input \
        --mount type=bind,source=/scratch/jchen/python_projects/TransMorph_brain_registration/test_output,target=/output \
        jchen245/transmorph_brain_mri_registration:transmorph_brain_mri_t1_v2

