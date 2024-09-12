#!/usr/bin/env bash
echo "Building docker image..."
docker build -f Dockerfile -t transmorph_brain_mri_t1 .
