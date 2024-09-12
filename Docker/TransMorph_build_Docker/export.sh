#!/usr/bin/env bash

bash ./build.sh

docker save transmorph_brain_mri_t1 | gzip -c > transmorph_brain_mri_t1.tar.gz
