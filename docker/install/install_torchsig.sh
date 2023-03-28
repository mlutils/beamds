#!/bin/bash


git clone https://github.com/TorchDSP/torchsig.git
cd torchsig/
pip install -r requirements.txt
pip install -U timm
pip install .
cd ../
