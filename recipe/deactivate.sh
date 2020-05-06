#!/bin/bash


if [[ $CONDA_CUPY_CUDA_PATH ]]; then
    export CUDA_PATH=$CONDA_CUPY_CUDA_PATH
    unset CONDA_CUPY_CUDA_PATH
else
    unset CUDA_PATH
fi
