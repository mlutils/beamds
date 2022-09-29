# follows https://docs.dgl.ai/install/index.html#install-from-source

git clone --recurse-submodules https://github.com/dmlc/dgl.git
cd dgl
mkdir build
cd build
cmake -DUSE_CUDA=ON ..
make -j4
cd ../python
python setup.py install
