#!/bin/bash

apt update
apt install -y swig
git clone https://github.com/facebookresearch/faiss.git
cd faiss/
cmake -B build -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_PYTHON=ON -DCMAKE_CUDA_ARCHITECTURES="90;89;86;75" .
make -C build -j faiss
make -C build -j swigfaiss
(cd build/faiss/python && python setup.py install)
cd ../