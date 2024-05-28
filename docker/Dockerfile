# Build Command:
# cd /path/to/beam_repo

# suggested beam names: spark flame blaze flash flare glow burst rush ray shine

#FROM nvidia/cuda:12.3.0-devel-ubuntu22.04
FROM nvcr.io/nvidia/pytorch:24.05-py3
LABEL "description"="beam image based on nvidia pytorch docker"
LABEL "run_command"="bash docker/run_beam.sh <image> <name> <initials-up-to-3-digits>"
LABEL "standalone_run_command"="docker run -p 28000-28099:28000-28099 --gpus=all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it -v /home/:/home/ -v /mnt/:/mnt/ --name <name> beam:<date> 280"
LABEL "build_command"="bash docker/build_beam.sh"

ENV DEBIAN_FRONTEND noninteractive

# pytorch geometric

ENV CPATH "/usr/local/cuda/include:$CPATH"
ENV PATH "/usr/local/cuda/bin:$PATH"
ENV LD_LIBRARY_PATH "/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

COPY docker/install /workspace/beam_image
RUN mkdir -p /workspace/beam_image

# apt installs
# COPY docker/install/apt_installs.sh /workspace/beam_image
RUN bash /workspace/beam_image/apt_installs.sh

# install docker
# COPY docker/install/install_docker.sh /workspace/beam_image
RUN bash /workspace/beam_image/install_docker.sh

# pip installs
RUN pip install --upgrade pip setuptools wheel

# COPY docker/install/requirements.txt /workspace/beam_image
RUN pip install -U -r /workspace/beam_image/requirements.txt

#RUN pip list
# COPY docker/install/requirements_0013.txt /workspace/beam_image
RUN pip install -U -r /workspace/beam_image/requirements_0013.txt -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
#RUN pip list
# COPY docker/install/requirements_0001.txt /workspace/beam_image
RUN pip install -U -r /workspace/beam_image/requirements_0001.txt
#RUN pip list
# COPY docker/install/requirements_0002.txt /workspace/beam_image
RUN pip install -U -r /workspace/beam_image/requirements_0002.txt
#RUN pip list
# COPY docker/install/requirements_0003.txt /workspace/beam_image
RUN pip install -U -r /workspace/beam_image/requirements_0003.txt
#RUN pip list
# COPY docker/install/requirements_0004.txt /workspace/beam_image
RUN pip install -U -r /workspace/beam_image/requirements_0004.txt
#RUN pip list
# COPY docker/install/requirements_0005.txt /workspace/beam_image
RUN pip install -U -r /workspace/beam_image/requirements_0005.txt
#RUN pip list
# COPY docker/install/requirements_0006.txt /workspace/beam_image
RUN pip install -U -r /workspace/beam_image/requirements_0006.txt
#RUN pip list
# COPY docker/install/requirements_0007.txt /workspace/beam_image
RUN pip install -U -r /workspace/beam_image/requirements_0007.txt
#RUN pip list
# COPY docker/install/requirements_0008.txt /workspace/beam_image
RUN pip install -U -r /workspace/beam_image/requirements_0008.txt
#RUN pip list
# COPY docker/install/requirements_0009.txt /workspace/beam_image
RUN pip install -r /workspace/beam_image/requirements_0009.txt
#RUN pip list
# COPY docker/install/requirements_0010.txt /workspace/beam_image
RUN pip install -U -r /workspace/beam_image/requirements_0010.txt
#RUN pip list
# COPY docker/install/requirements_0011.txt /workspace/beam_image
RUN pip install -U -r /workspace/beam_image/requirements_0011.txt
#RUN pip list
# COPY docker/install/requirements_0012.txt /workspace/beam_image
RUN pip install -U -r /workspace/beam_image/requirements_0012.txt
#RUN pip list
# COPY docker/install/requirements_0014.txt /workspace/beam_image
RUN pip install -U -r /workspace/beam_image/requirements_0014.txt
#RUN pip list
# COPY docker/install/requirements_0015.txt /workspace/beam_image
RUN pip install -U -r /workspace/beam_image/requirements_0015.txt
# RUN pip list
# COPY docker/install/requirements_0016.txt /workspace/beam_image
RUN pip install -U -r /workspace/beam_image/requirements_0016.txt
# RUN pip list
# COPY docker/install/requirements_0017.txt /workspace/beam_image
RUN pip install -U -r /workspace/beam_image/requirements_0017.txt
# RUN pip list
# COPY docker/install/requirements_0018.txt /workspace/beam_image
RUN pip install -U -r /workspace/beam_image/requirements_0018.txt
#install huggingface ecosystem
# REMOVE FOR LITE VERSION
# COPY docker/install/install_huggingface_ecosystem.sh /workspace/beam_image
RUN bash /workspace/beam_image/install_huggingface_ecosystem.sh

# COPY docker/install/requirements_no_deps.txt /workspace/beam_image
RUN pip install -U --no-deps -r /workspace/beam_image/requirements_no_deps.txt
#RUN pip list
# COPY docker/install/verify_requirements.txt /workspace/beam_image
RUN pip install -U --no-deps -r /workspace/beam_image/verify_requirements.txt

#install pytorch geometric
# REMOVE FOR LITE VERSION
# COPY docker/install/install_pyg.sh /workspace/beam_image
RUN bash /workspace/beam_image/install_pyg.sh

#install torchsig
# REMOVE FOR LITE VERSION
# COPY docker/install/install_torchsig.sh /workspace/beam_image
RUN bash /workspace/beam_image/install_torchsig.sh

# install DGL from source
# REMOVE FOR LITE VERSION
RUN bash /workspace/install/install_dgl.sh
# COPY docker/install_dgl_fixed.sh /workspace/install/install_dgl_fixed.sh
# RUN bash /workspace/install/install_dgl_fixed.sh


# install RAPID framework
# cannot install rapids as it does not support cuda 12.0 yet
# see https://rapids.ai/pip.html#install
# RUN bash /workspace/beam_image/rapid_install.sh

# download datasets and models
# REMOVE FOR LITE VERSION
# RUN python /workspace/beam_image/downloader.py

## download nlp data
# COPY docker/install/models_downloads.sh /workspace/beam_image
RUN bash /workspace/beam_image/models_downloads.sh


# no need to install RAPIDS as it comes with the image, except cusignal which is not found
# REMOVE FOR LITE VERSION
# COPY docker/install/install_cusignal.sh /workspace/beam_image
RUN bash /workspace/beam_image/install_cusignal.sh


# RUN pip install faiss-gpu
# REMOVE FOR LITE VERSION
# COPY docker/install/install_faiss.sh /workspace/beam_image
RUN bash /workspace/beam_image/install_faiss.sh

# Install cx_oracle
ENV ORACLE_HOME "/opt/oracle/instantclient"
ENV LD_LIBRARY_PATH "$ORACLE_HOME:$LD_LIBRARY_PATH"


#COPY docker/install/install_oracle.sh /workspace/beam_image
RUN bash /workspace/beam_image/install_oracle.sh

# install beam - from pip (see requirements_no_deps.txt file)
#COPY docker/install/install_beam.sh /workspace/beam_image
#RUN bash /workspace/beam_image/install_beam.sh
#ENV PATH "$PATH:/workspace/beamds/notebooks/"

# install odbc driver
#COPY docker/install/install_odbc.sh /workspace/beam_image
RUN bash /workspace/beam_image/install_odbc.sh

# install hadoop
#COPY docker/install/install_hadoop.sh /workspace/beam_image
RUN bash /workspace/beam_image/install_hadoop.sh

# Set Hadoop environment variables
ENV HADOOP_HOME "/usr/local/hadoop"
ENV LD_LIBRARY_PATH "$LD_LIBRARY_PATH:$HADOOP_HOME/lib/native"
ENV PATH "$PATH:$HADOOP_HOME/bin"
# set default java version
ENV JAVA_HOME "/usr/lib/jvm/default-java"

# config jupyter notebook
#COPY docker/install/configure_jupyter.sh /workspace/beam_image
RUN bash /workspace/beam_image/configure_jupyter.sh

# install language servers
#COPY docker/install/language_servers.sh /workspace/beam_image
RUN bash /workspace/beam_image/language_servers.sh

# install aux installs
#COPY docker/install/aux_installs.sh /workspace/beam_image
RUN bash /workspace/beam_image/aux_installs.sh

RUN bash /workspace/beam_image/usermode_ssh_server.sh

COPY docker/runs /workspace/beam_image/runs
# COPY docker/install/entrypoint.sh /workspace/beam_image

RUN chmod +x /workspace/beam_image/entrypoint.sh
ENTRYPOINT ["/workspace/beam_image/entrypoint.sh"]
CMD ["10"]


