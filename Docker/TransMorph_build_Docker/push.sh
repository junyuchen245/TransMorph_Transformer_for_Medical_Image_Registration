#!/usr/bin/env bash

bash ./build.sh

docker tag transmorph_brain_mri_t1 jchen245/transmorph_brain_mri_registration:transmorph_brain_mri_t1_v0

docker push jchen245/transmorph_brain_mri_registration:transmorph_brain_mri_t1_v0
