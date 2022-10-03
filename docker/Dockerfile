# Build Command:
# cd /path/to/beam_repo

# suggested beam names: spark flame blaze flash flare glow burst rush ray shine

FROM nvcr.io/nvidia/pytorch:22.08-py3
LABEL "description"="beam image based on nvidia pytorch docker"
LABEL "run_command"="bash docker/run_beam.sh <image> <name> <initials-up-to-3-digits>"
LABEL "standalone_run_command"="docker run -p 28000-28099:28000-28099 --gpus=all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it -v /home/:/home/ -v /mnt/:/mnt/ --name <name> beam:<date> 280"
LABEL "build_command"="bash docker/build_beam.sh"

ENV DEBIAN_FRONTEND noninteractive

COPY docker/install /workspace/install

# pytorch geometric

#export CPATH="/usr/local/cuda/include:$CPATH"
#export PATH="/usr/local/cuda/bin:$PATH"
#export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
#export DYLD_LIBRARY_PATH="/usr/local/cuda/lib:$DYLD_LIBRARY_PATH"

ENV CPATH "/usr/local/cuda/include:$CPATH"
ENV PATH "/usr/local/cuda/bin:$PATH"
ENV LD_LIBRARY_PATH "/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
ENV DYLD_LIBRARY_PATH "/usr/local/cuda/lib:$DYLD_LIBRARY_PATH"

# apt installs
#RUN chmod +x /workspace/install/apt_installs.sh
RUN bash /workspace/install/apt_installs.sh

# pip installs
RUN pip install --upgrade pip
RUN pip install -U -r /workspace/install/requirements.txt
RUN pip install -U -r /workspace/install/requirements_torch_geometric.txt
RUN pip install -U -r /workspace/install/requirements_0001.txt
RUN pip install -U -r /workspace/install/requirements_0002.txt
RUN pip install -U -r /workspace/install/requirements_0003.txt
RUN pip install -U -r /workspace/install/requirements_0004.txt
RUN pip install -U -r /workspace/install/requirements_0005.txt
RUN pip install -U -r /workspace/install/requirements_0006.txt
RUN pip install -U -r /workspace/install/requirements_0007.txt
RUN pip install -U -r /workspace/install/requirements_0008.txt
RUN pip install -U -r /workspace/install/requirements_0009.txt
RUN pip install -U -r /workspace/install/requirements_0010.txt
RUN pip install -U -r /workspace/install/requirements_0011.txt
RUN pip install -U -r /workspace/install/requirements_0012.txt
RUN pip install -U -r /workspace/install/requirements_0013.txt -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install -U --no-deps -r /workspace/install/requirements_no_deps.txt
RUN pip install -U --no-deps -r /workspace/install/verify_requirements.txt

# Install faiss if you encounter GPU issues when using the pip version:
# https://github.com/kyamagu/faiss-wheels/issues/54
RUN conda install -y -c conda-forge faiss-gpu

# install DGL from source
RUN bash /workspace/install/install_dgl.sh

# add the beam message

RUN cp /workspace/install/motd /etc/motd
RUN echo "cat /etc/motd" >> /root/.bashrc

# download datasets and models
RUN python /workspace/install/downloader.py

## download nlp data

RUN python -m nltk.downloader all

#ENV LC_ALL C.UTF-8
#ENV LANG C.UTF-8
#RUN python -m spacy download zh_core_web_md
#RUN python -m spacy download en_core_web_md
#RUN python -m spacy download xx_ent_wiki_sm
#RUN python -m spacy download ru_core_news_sm

# install beam

RUN mkdir /workspace/beamds/
COPY . /workspace/beamds/
WORKDIR /workspace/beamds/
RUN python -m build
RUN pip install "dist/beam_ds-$(python src/beam/_version.py)-py3-none-any.whl"
WORKDIR /workspace

# config jupyter notebook
RUN jupyter nbextension enable --py widgetsnbextension

RUN jupyter lab --generate-config
RUN echo "c.ServerApp.notebook_dir = '/home/'" >> /root/.jupyter/jupyter_lab_config.py
RUN echo "c.ServerApp.allow_remote_access = True" >> /root/.jupyter/jupyter_lab_config.py
RUN echo "c.ServerApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_lab_config.py
RUN echo "c.ServerApp.token = ''" >> /root/.jupyter/jupyter_lab_config.py
RUN echo "c.ServerApp.allow_root = True" >> /root/.jupyter/jupyter_lab_config.py


# for now this line throw unclear error when running in the Dockerfile, but it runs fine from the container
# so for now, we will run it from the container and commit the container to a modified image.
# RUN pip install git+https://github.com/chaoleili/jupyterlab_tensorboard.git

RUN chmod +x /workspace/install/entrypoint.sh
ENTRYPOINT ["/workspace/install/entrypoint.sh"]
CMD ["10"]


