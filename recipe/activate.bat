@echo off
if defined CONDA_BUILD_STATE (
    @echo on
)

if defined CUDA_PATH (
    set "CONDA_CUPY_CUDA_PATH=%CUDA_PATH%"
)

set "CUDA_PATH=%CONDA_PREFIX%"
