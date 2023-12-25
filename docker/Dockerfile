# Build Command:
# cd /path/to/beam_repo

# suggested beam names: spark flame blaze flash flare glow burst rush ray shine

FROM nvcr.io/nvidia/pytorch:23.12-py3
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

# after failure use this to aviod restarting the build process
# COPY docker/install_fixed/requirements_0016.txt /workspace/install/requirements_0016.txt

# apt installs
RUN bash /workspace/install/apt_installs.sh
#RUN pip list

# install docker
RUN bash /workspace/install/install_docker.sh

# pip installs
RUN pip install --upgrade pip setuptools wheel
RUN pip install -U -r /workspace/install/requirements.txt

#RUN pip list
RUN pip install -U -r /workspace/install/requirements_0013.txt -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
#RUN pip list
RUN pip install -U -r /workspace/install/requirements_0001.txt
#RUN pip list
RUN pip install -U -r /workspace/install/requirements_0002.txt
#RUN pip list
RUN pip install -U -r /workspace/install/requirements_0003.txt
#RUN pip list
# COPY docker/install_fixed/requirements_0004.txt /workspace/install/requirements_0004.txt
RUN pip install -U -r /workspace/install/requirements_0004.txt
#RUN pip list
# COPY docker/install_fixed/requirements_0005.txt /workspace/install/requirements_0005.txt
RUN pip install -U -r /workspace/install/requirements_0005.txt
#RUN pip list
RUN pip install -U -r /workspace/install/requirements_0006.txt
#RUN pip list
RUN pip install -U -r /workspace/install/requirements_0007.txt
#RUN pip list
RUN pip install -U -r /workspace/install/requirements_0008.txt
#RUN pip list
RUN pip install -r /workspace/install/requirements_0009.txt
#RUN pip list
RUN pip install -U -r /workspace/install/requirements_0010.txt
#RUN pip list
RUN pip install -U -r /workspace/install/requirements_0011.txt
#RUN pip list
RUN pip install -U -r /workspace/install/requirements_0012.txt
#RUN pip list
RUN pip install -U -r /workspace/install/requirements_0014.txt
#RUN pip list
RUN pip install -U -r /workspace/install/requirements_0015.txt
# RUN pip list

RUN pip install -U -r /workspace/install/requirements_0016.txt

RUN pip install -U -r /workspace/install/requirements_0017.txt
#COPY docker/install_fixed/requirements_fixed.txt /workspace/install/requirements_fixed.txt
#RUN pip install -U -r /workspace/install/requirements_fixed.txt

# COPY docker/install_fixed/requirements_no_deps.txt /workspace/install/requirements_no_deps.txt
RUN pip install -U --no-deps -r /workspace/install/requirements_no_deps.txt
#RUN pip list
RUN pip install -U --no-deps -r /workspace/install/verify_requirements.txt
RUN pip list

#install pytorch geometric
# REMOVE FOR LITE VERSION
RUN bash /workspace/install/install_pyg.sh

#install torchsig
# REMOVE FOR LITE VERSION
RUN bash /workspace/install/install_torchsig.sh

# install DGL from source
# REMOVE FOR LITE VERSION
RUN bash /workspace/install/install_dgl.sh

# install RAPID framework
# cannot install rapids as it does not support cuda 12.0 yet
# see https://rapids.ai/pip.html#install
# RUN bash /workspace/install/rapid_install.sh

# download datasets and models
# REMOVE FOR LITE VERSION
# RUN python /workspace/install/downloader.py

## download nlp data
# RUN python /workspace/install/models_downloads.py


# no need to install RAPIDS as it comes with the image, except cusignal which is not found
# REMOVE FOR LITE VERSION
RUN bash /workspace/install/install_cusignal.sh


# RUN pip install faiss-gpu
# REMOVE FOR LITE VERSION
RUN bash /workspace/install/install_faiss.sh

# Install cx_oracle
ENV ORACLE_HOME /opt/oracle/instantclient
ENV LD_LIBRARY_PATH $ORACLE_HOME:$LD_LIBRARY_PATH
RUN bash /workspace/install/install_oracle.sh

# install beam - from pip (see requirements_no_deps.txt file)
RUN bash /workspace/install/install_beam.sh
ENV PATH "$PATH:/workspace/beamds/notebooks/"

# install odbc driver
RUN bash /workspace/install/install_odbc.sh

# install hadoop
RUN bash /workspace/install/install_hadoop.sh
# Set Hadoop environment variables
ENV HADOOP_HOME=/usr/local/hadoop
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HADOOP_HOME/lib/native
ENV PATH=$PATH:$HADOOP_HOME/bin

# set default java version
ENV JAVA_HOME /usr/lib/jvm/default-java

# config jupyter notebook
RUN bash /workspace/install/configure_jupyter.sh
RUN bash /workspace/install/language_servers.sh


RUN bash /workspace/install/aux_installs.sh


# for now this line throw unclear error when running in the Dockerfile, but it runs fine from the container
# so for now, we will run it from the container and commit the container to a modified image.
# RUN pip install git+https://github.com/chaoleili/jupyterlab_tensorboard.git

RUN chmod +x /workspace/install/entrypoint.sh
ENTRYPOINT ["/workspace/install/entrypoint.sh"]
CMD ["10"]


