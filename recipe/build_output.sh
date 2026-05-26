#!/bin/bash

set -ex

# Debugging aarch64 build failures
find $BUILD_PREFIX -name "cusparseLt.h"
find $PREFIX -name "cusparseLt.h"

export NVCC="$(which nvcc)"
if [ "${CUDA_MAJOR}" -ge 12 ]; then
    export CUDA_PATH=$PREFIX/targets/${TARGET_NAME}
fi
echo "nvcc is $NVCC, CUDA path is $CUDA_PATH, CUDA major version is $CUDA_MAJOR"
echo "CONDA_OVERRIDE_CUDA is $CONDA_OVERRIDE_CUDA"

${PYTHON} -m pip install . --no-deps -vv

# rattler-build does not add 't' suffix to SP_DIR for freethreading python as
# expected. Manually adding 't' suffix as a workaround (solution found in the
# following URL:
#   - https://github.com/conda/conda-build/issues/5563#issuecomment-3996514657
#
# Can (hopefully) be removed once this ticket is resolved:
#   - https://github.com/prefix-dev/rattler-build/issues/2404
export SP_DIR=$PREFIX/lib/python`python -c "import sysconfig; print(sysconfig.get_config_var('LDVERSION'))"`/site-packages
mkdir -p "${SP_DIR}/cupy/.data/"
cp ${RECIPE_DIR}/preload_config/linux64_cuda${CUDA_MAJOR}_wheel.json ${SP_DIR}/cupy/.data/_wheel.json
