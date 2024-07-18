# follows https://docs.dgl.ai/install/index.html#install-from-source

# RUN pip install  dgl -f https://data.dgl.ai/wheels/repo.html
# RUN pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html

# Install dgl from source so it would be compatible with cuda 12.0
git clone --recurse-submodules https://github.com/dmlc/dgl.git
cd dgl
mkdir build
cd build
cmake -DUSE_CUDA=ON ..
make -j4
cd ../python
python setup.py install
