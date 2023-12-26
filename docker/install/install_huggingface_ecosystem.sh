# Get the CUDA version installed on the system
cuda_version_installed=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')

# Extract the major and minor version numbers
cuda_major=$(echo $cuda_version_installed | cut -d. -f1)
cuda_minor=$(echo $cuda_version_installed | cut -d. -f2)

# Standardize the CUDA version to three digits
MY_CUDA_VERSION="${cuda_major}${cuda_minor}"

# Format the make argument
MY_MAKE_ARGUMENT="cuda${cuda_major}x"

# see: https://github.com/TimDettmers/bitsandbytes
git clone https://github.com/timdettmers/bitsandbytes.git
cd bitsandbytes

# CUDA_VERSIONS in {110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 120}
# make argument in {cuda110, cuda11x, cuda12x}
# if you do not know what CUDA you have, try looking at the output of: python -m bitsandbytes
CUDA_VERSION=${MY_CUDA_VERSION} make ${MY_MAKE_ARGUMENT}
python setup.py install


pip install safetensors accelerate deepspeed
# vllm