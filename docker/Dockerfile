# Build Command:
# cd /path/to/beam_repo

# suggested beam names: spark flame blaze flash flare glow burst rush ray shine

FROM nvcr.io/nvidia/pytorch:23.06-py3
LABEL "description"="beam image based on nvidia pytorch docker"
LABEL "run_command"="bash docker/run_beam.sh <image> <name> <initials-up-to-3-digits>"
LABEL "standalone_run_command"="docker run -p 28000-28099:28000-28099 --gpus=all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it -v /home/:/home/ -v /mnt/:/mnt/ --name <name> beam:<date> 280"
LABEL "build_command"="bash docker/build_beam.sh"

ENV DEBIAN_FRONTEND noninteractive

COPY docker/install /workspace/install

# pytorch geometric

ENV CPATH "/usr/local/cuda/include:$CPATH"
ENV PATH "/usr/local/cuda/bin:$PATH"
ENV LD_LIBRARY_PATH "/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

# only needed for macOS platforms
#ENV DYLD_LIBRARY_PATH "/usr/local/cuda/lib:$DYLD_LIBRARY_PATH"

# apt installs
#RUN chmod +x /workspace/install/apt_installs.sh
RUN bash /workspace/install/apt_installs.sh
RUN pip list

# pip installs
RUN pip install --upgrade pip
RUN pip install -U -r /workspace/install/requirements.txt
# REMOVE FOR LITE VERSION
RUN pip install -U -r /workspace/install/requirements_torch_geometric.txt
RUN pip list
RUN pip install -U -r /workspace/install/requirements_0013.txt -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip list
RUN pip install -U -r /workspace/install/requirements_0001.txt
RUN pip list
RUN pip install -U -r /workspace/install/requirements_0002.txt
RUN pip list
RUN pip install -U -r /workspace/install/requirements_0003.txt
RUN pip list
RUN pip install -U -r /workspace/install/requirements_0004.txt
RUN pip list
RUN pip install -U -r /workspace/install/requirements_0005.txt
RUN pip list
RUN pip install -U -r /workspace/install/requirements_0006.txt
RUN pip list
RUN pip install -U -r /workspace/install/requirements_0007.txt
RUN pip list
RUN pip install -U -r /workspace/install/requirements_0008.txt
RUN pip list
RUN pip install -r /workspace/install/requirements_0009.txt
RUN pip list
RUN pip install -U -r /workspace/install/requirements_0010.txt
RUN pip list
RUN pip install -U -r /workspace/install/requirements_0011.txt
RUN pip list
RUN pip install -U -r /workspace/install/requirements_0012.txt
RUN pip list
RUN pip install -U -r /workspace/install/requirements_0014.txt
RUN pip list
RUN pip install -U --no-deps -r /workspace/install/requirements_no_deps.txt
RUN pip list
RUN pip install -U --no-deps -r /workspace/install/verify_requirements.txt
RUN pip list

#install torchsig
RUN bash /workspace/install/install_torchsig.sh

# Install dgl from source so it would be compatible with cuda 12.0
# RUN pip install  dgl -f https://data.dgl.ai/wheels/repo.html
# RUN pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html

# install DGL from source
# REMOVE FOR LITE VERSION
RUN bash /workspace/install/install_dgl.sh

# Install faiss if you encounter GPU issues when using the pip version:
# https://github.com/kyamagu/faiss-wheels/issues/54

# install RAPID framework
# cannot install rapids as it does not support cuda 12.0 yet
# see https://rapids.ai/pip.html#install
# RUN bash /workspace/install/rapid_install.sh

# download datasets and models
# REMOVE FOR LITE VERSION
RUN python /workspace/install/downloader.py

## download nlp data
# RUN python /workspace/install/models_downloads.pyin

# REMOVE FOR LITE VERSION

# no need to install RAPIDS as it comes with the image, except cusignal which is not found
RUN bash /workspace/install/install_cusignal.sh
# RUN pip install cudf-cu11 dask-cudf-cu11 --extra-index-url=https://pypi.ngc.nvidia.com
# RUN pip install cuml-cu11 --extra-index-url=https://pypi.ngc.nvidia.com
# RUN pip install cugraph-cu11 --extra-index-url=https://pypi.ngc.nvidia.com


# need to see how to install faiss-gpu. for now we install only the cpu version
# RUN conda install -y -c conda-forge faiss-gpu
# RUN pip install faiss-gpu
RUN bash /workspace/install/install_faiss.sh

# install beam - from pip (see requirements_no_deps.txt file)

RUN mkdir /workspace/beamds/
COPY . /workspace/beamds/
ENV PATH "$PATH:/workspace/beamds/notebooks/"

#WORKDIR /workspace/beamds/
#RUN python -m build
#RUN pip install "dist/beam_ds-$(python src/beam/_version.py)-py3-none-any.whl"
#WORKDIR /workspace

# install odbc driver
# https://learn.microsoft.com/en-us/sql/connect/odbc/linux-mac/installing-the-microsoft-odbc-driver-for-sql-server?view=sql-server-ver16&tabs=ubuntu18-install%2Calpine17-install%2Cdebian8-install%2Credhat7-13-install%2Crhel7-offline
RUN bash /workspace/install/install_odbc.sh

# config jupyter notebook
RUN bash /workspace/install/configure_jupyter.sh
# RUN bash /workspace/install/language_servers.sh

RUN bash /workspace/install/aux_installs.sh

# for now this line throw unclear error when running in the Dockerfile, but it runs fine from the container
# so for now, we will run it from the container and commit the container to a modified image.
# RUN pip install git+https://github.com/chaoleili/jupyterlab_tensorboard.git

RUN chmod +x /workspace/install/entrypoint.sh
ENTRYPOINT ["/workspace/install/entrypoint.sh"]
CMD ["10"]


