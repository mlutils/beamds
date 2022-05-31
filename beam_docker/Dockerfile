# Build Command:
# cd /path/to/beam_repo
# docker build -f beam_docker/Dockerfile --tag beam:$(date '+%Y%m%d') .
# Run command:
# docker run -p 12000-12999:12000-12999 --gpus=all --ipc=host -itd -v /home/:/home/ --name flame beam:<date> 12

# suggested beam names: spark flame blaze flash flare glow burst rush ray shine

FROM nvcr.io/nvidia/pytorch:22.03-py3
LABEL "description"="beam image based on nvidia pytorch docker"
ENV DEBIAN_FRONTEND noninteractive

COPY beam_docker/install /workspace/install

# apt installs
RUN chmod +x /workspace/install/apt_installs.sh
RUN /workspace/install/apt_installs.sh

# pip installs
RUN chmod +x /workspace/install/pip_installs.sh
RUN /workspace/install/pip_installs.sh

# add the beam message

RUN cp /workspace/install/motd /etc/motd
RUN echo "cat /etc/motd" >> /root/.bashrc

# download datasets and models
RUN python /workspace/install/downloader.py

# pytorch geometric

ENV CPATH "/usr/local/cuda/include:$CPATH"
ENV PATH "/usr/local/cuda/bin:$PATH"

ENV LD_LIBRARY_PATH "/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
ENV DYLD_LIBRARY_PATH "/usr/local/cuda/lib:$DYLD_LIBRARY_PATH"


RUN pip install torch-scatter torch-sparse torch-geometric torch-spline-conv torch-cluster

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
CMD ["10022", "10088", "10123456"]


