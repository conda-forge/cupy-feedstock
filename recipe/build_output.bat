for /f "tokens=* usebackq" %%f in (`where nvcc`) do (
    set "dummy=%%f"
    call set "NVCC=%%dummy:\=\\%%"
)

set "NVCC=%NVCC% --use-local-env"
set "CUDA_PATH=%PREFIX%\Library"
echo "nvcc is %NVCC%, CUDA path is %CUDA_PATH%"
echo "CONDA_OVERRIDE_CUDA is %CONDA_OVERRIDE_CUDA%"

%PYTHON% -m pip install . --no-deps -vv
if errorlevel 1 exit 1

if not exist %SP_DIR%\cupy\.data mkdir %SP_DIR%\cupy\.data
copy %RECIPE_DIR%\preload_config\win64_cuda%CUDA_MAJOR%_wheel.json %SP_DIR%\cupy\.data\_wheel.json
if not exist %SP_DIR%\cupy\.data\_depends.json exit 1
