#!/bin/bash

# this is the old way of installing rapids
# RUN pip install cudf-cu11 dask-cudf-cu11 --extra-index-url=https://pypi.ngc.nvidia.com
# RUN pip install cuml-cu11 --extra-index-url=https://pypi.ngc.nvidia.com
# RUN pip install cugraph-cu11 --extra-index-url=https://pypi.ngc.nvidia.com

export PREFIX=/usr/local
export CUSIGNAL_HOME=/workspace/cusignal
git clone https://github.com/rapidsai/cusignal.git $CUSIGNAL_HOME
cd $CUSIGNAL_HOME
./build.sh
cd /workspace
