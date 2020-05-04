#!/bin/bash


if [[ -z $CUDA_PATH ]]; then
    export CONDA_CUPY_CUDA_PATH=$CUDA_PATH
fi

export CUDA_PATH=$CONDA_PREFIX
