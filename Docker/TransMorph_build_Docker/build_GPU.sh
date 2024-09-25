#!/usr/bin/env bash
echo "Building docker image..."
docker build -f Dockerfile_GPU -t transmorph_brain_mri_t1_gpu .
