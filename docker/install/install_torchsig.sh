#!/bin/bash


git clone https://github.com/TorchDSP/torchsig.git
cd torchsig/
#pip install -r requirements.txt
#pip install -U timm
pip install -U h5py numba ipdb PyWavelets lmdb gdown icecream sympy torchmetrics click
pip install --no-deps .
cd ../
