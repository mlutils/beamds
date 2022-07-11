# Build Command:
# cd /path/to/beam_repo

# suggested beam names: spark flame blaze flash flare glow burst rush ray shine

FROM nvcr.io/nvidia/pytorch:22.06-py3
LABEL "description"="beam image based on nvidia pytorch docker"
LABEL "run_command"="docker run -p 12000-12099:12000-12099 --gpus=all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -itd -v /home/:/home/ --name flame beam:<date> 12"
LABEL "build_command"="docker build -f docker/Dockerfile --tag beam:$(date '+%Y%m%d') ."

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
RUN chmod +x /workspace/install/apt_installs.sh
RUN /workspace/install/apt_installs.sh

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
RUN pip install -U --no-deps -r /workspace/install/requirements_no_deps.txt
RUN pip install -U --no-deps -r /workspace/install/verify_requirements.txt


# add the beam message

RUN cp /workspace/install/motd /etc/motd
RUN echo "cat /etc/motd" >> /root/.bashrc

# download datasets and models
RUN python /workspace/install/downloader.py

## download nlp data
RUN python -m nltk.downloader all
RUN python -m spacy download zh_core_web_sm
RUN python -m spacy download en_core_web_sm
RUN python -m spacy download xx_ent_wiki_sm
RUN python -m spacy download ru_core_news_sm

# install beam

RUN mkdir /workspace/beamds/
COPY . /workspace/beamds/
WORKDIR /workspace/beamds/
RUN python -m build
RUN pip install --no-deps "dist/beam-$(python -c 'import src.beam as beam; print(beam.__version__)')-py3-none-any.whl"
WORKDIR /workspace

# config jupyter notebook
RUN jupyter nbextension enable --py widgetsnbextension

RUN jupyter lab --generate-config
RUN echo "c.ServerApp.notebook_dir = '/home/'" >> /root/.jupyter/jupyter_lab_config.py
RUN echo "c.ServerApp.allow_remote_access = True" >> /root/.jupyter/jupyter_lab_config.py
RUN echo "c.ServerApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_lab_config.py
RUN echo "c.ServerApp.token = ''" >> /root/.jupyter/jupyter_lab_config.py
RUN echo "c.ServerApp.allow_root = True" >> /root/.jupyter/jupyter_lab_config.py

RUN pip install git+https://github.com/cliffwoolley/jupyter_tensorboard.git
RUN pip install git+https://github.com/chaoleili/jupyterlab_tensorboard.git

RUN chmod +x /workspace/install/entrypoint.sh
ENTRYPOINT ["/workspace/install/entrypoint.sh"]
CMD ["10"]


