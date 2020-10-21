# Configure CuPy to use 1 GPU for testing
import os
os.environ["CUPY_TEST_GPU_LIMIT"] = "1"

# Check CUDA_PATH is set
cuda_path = os.environ.get('CUDA_PATH')
assert cuda_path is not None
print("CUDA_PATH:", cuda_path)

# Now that conda-forge docker images have libcuda.so, so "import cupy" would not fail.
# However, tests would fail on the Azure CI since there is no GPU. See the discussion
# in https://github.com/conda-forge/cupy-feedstock/pull/59#issuecomment-629584090
import sys
import cupy
from cupy_backends.cuda.api.driver import get_build_version
if get_build_version() in (11000,):
    config_json = os.path.dirname(cupy.__file__) + '/.data/_wheel.json'
    if not os.path.isfile(config_json):
        raise RuntimeError('_wheel.json is not found in the package')
    else:
        print("_wheel.json is found:", config_json)
try:
    # Print CuPy runtime info
    # this line would fail if there is no GPU
    cupy.show_config()
except Exception as e:
    print('Got an error: \n%s' % str(e))
    print("No GPU available. Exiting without running CuPy's tests.")
    sys.exit(0)

# Run CuPy's test suite
import py
py.test.cmdline.main(["tests/cupy_tests"])
py.test.cmdline.main(["tests/cupyx_tests"])
