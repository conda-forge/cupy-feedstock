@echo off
if defined CONDA_BUILD_STATE (
    @echo on
)

if not "%CONDA_CUPY_CUDA_PATH%"=="" (
    set "CUDA_PATH=%CONDA_CUPY_CUDA_PATH%"
    set "CONDA_CUPY_CUDA_PATH="
) else (
    set CUDA_PATH=""
)
