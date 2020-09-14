#!/bin/bash


if [[ -n ${CONDA_CUPY_CUDA_PATH:+x} ]]; then
    export CUDA_PATH=$CONDA_CUPY_CUDA_PATH
    unset CONDA_CUPY_CUDA_PATH
else
    unset CUDA_PATH
fi
