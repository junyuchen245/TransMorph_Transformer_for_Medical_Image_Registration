#!/usr/bin/env bash
bash ./build.sh
#jchen245/transmorph_brain_mri_registration:transmorph_brain_mri_t1_v0
#-it --entrypoint "/bin/bash"
docker run --rm \
        --ipc=host \
        --memory 256g \
        --gpus "device=0"\
        --mount type=bind,source=/scratch/jchen/python_projects/TransMorph_brain_registration/test_dataset_monkey.json,target=/app/input_dataset.json \
        --mount type=bind,source=/scratch/jchen/python_projects/TransMorph_brain_registration/configs_registration.json,target=/app/configs_registration.json \
        --mount type=bind,source=/scratch/jchen/python_projects/TransMorph_brain_registration/test_data,target=/app/input \
        --mount type=bind,source=/scratch/jchen/python_projects/TransMorph_brain_registration/test_output,target=/app/output \
        transmorph_brain_mri_t1_gpu
