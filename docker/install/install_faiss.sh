#!/bin/bash

# Install faiss if you encounter GPU issues when using the pip version:
# https://github.com/kyamagu/faiss-wheels/issues/54

# need to see how to install faiss-gpu. for now we install only the cpu version
# RUN conda install -y -c conda-forge faiss-gpu

apt update
apt install -y swig
git clone https://github.com/facebookresearch/faiss.git
cd faiss/

# DCMAKE_CUDA_ARCHITECTURES:
# NVIDIA A100: Compute Capability 8.0 (Ampere architecture)
# NVIDIA V100: Compute Capability 7.0 (Volta architecture)
# RTX 8000 and RTX 6000: Both are based on the Turing architecture, which corresponds to Compute Capability 7.5
# RTX 3090 Ti: This is also part of the Ampere generation, similar to the A100, but with Compute Capability 8.6
# RTX 2080 Ti: This GPU has a compute capability of 7.5 (Turing architecture)

# The Compute Capability 9.0 (CC 9.0) is associated with NVIDIA's Hopper GPU family, specifically the H100 model.
# The Ada Lovelace GPUs, including the RTX 4090, are classified under Compute Capability 8.9 (CC 8.9).

cmake -B build -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_PYTHON=ON -DCMAKE_CUDA_ARCHITECTURES="80;75;86" .
make -C build -j faiss
make -C build -j swigfaiss
(cd build/faiss/python && python setup.py install)
cd ../