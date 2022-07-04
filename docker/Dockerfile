# Build Command:
# cd /path/to/beam_repo

# suggested beam names: spark flame blaze flash flare glow burst rush ray shine

FROM nvcr.io/nvidia/pytorch:22.04-py3
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
RUN pip install -U -r /workspace/install/requirements.txt
RUN pip install -U -r /workspace/install/requirements.txt
RUN pip install -U -r /workspace/install/requirements.txt
RUN pip install -U -r /workspace/install/requirements.txt
RUN pip install -U -r /workspace/install/requirements.txt
# RUN chmod +x /workspace/install/pip_installs.sh
# RUN /workspace/install/pip_installs.sh

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

# path for pycharm data
RUN mkdir /root/pycharm/

# install beam

RUN mkdir /workspace/beamds/
COPY . /workspace/beamds/
RUN python -m pip install --upgrade build
WORKDIR /workspace/beamds/
RUN python -m build
RUN pip install dist/*.whl --force-reinstall
WORKDIR /workspace

# config jupyter notebook

RUN jupyter nbextension enable --py widgetsnbextension

RUN jupyter notebook --generate-config
RUN echo "c.NotebookApp.notebook_dir = '/home/'" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.allow_remote_access = True" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.token = ''" >> /root/.jupyter/jupyter_notebook_config.py

RUN chmod +x /workspace/install/entrypoint.sh
ENTRYPOINT ["/workspace/install/entrypoint.sh"]
CMD ["10"]


