#!/bin/bash

export NVCC="$(which nvcc)"
if [ "${CUDA_MAJOR}" -ge 12 ]; then
    export CUDA_PATH=$PREFIX/targets/${TARGET_NAME}
fi
echo "nvcc is $NVCC, CUDA path is $CUDA_PATH, CUDA major version is $CUDA_MAJOR"
echo "CONDA_OVERRIDE_CUDA is $CONDA_OVERRIDE_CUDA"

${PYTHON} -m pip install . --no-deps -vv

mkdir -p "${SP_DIR}/cupy/.data/"
cp ${RECIPE_DIR}/preload_config/linux64_cuda${CUDA_MAJOR}_wheel.json ${SP_DIR}/cupy/.data/_wheel.json
