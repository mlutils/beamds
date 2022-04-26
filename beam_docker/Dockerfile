# Build Command:
# cd /path/to/beam_repo
# docker build -f beam_docker/Dockerfile --tag beam:$(date '+%Y%m%d') .
# Run command:
# docker run -p 12000-12999:12000-12999 --gpus=all --ipc=host -itd -v /home/:/home/ --name flame beam:<date> 12022 12088

# suggested beam names: spark flame blaze flash flare glow burst rush ray shine

FROM nvcr.io/nvidia/pytorch:22.03-py3
LABEL "description"="beam image based on nvidia pytorch docker"

RUN apt update

# ssh server
RUN apt install -y openssh-server
RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config

# applications
RUN apt install -y htop -y tmux -y alien -y libtool -y dh-autoreconf -y debhelper -y libaiol -y libaio-dev
RUN apt install -y telnet -y iputils-ping -y python-dev


RUN apt install -y graphviz
RUN apt install -y python3-dev graphviz pkg-config
RUN DEBIAN_FRONTEND=noninteractive apt install -y expect
RUN DEBIAN_FRONTEND=noninteractive apt install -y libgraphviz-dev
RUN apt install -y cron


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

# additional python packages

RUN pip install elasticsearch fasttext pandarallel gensim umap-learn pyvis graphviz pygraphviz opencv-python
RUN python -m pip install "dask[complete]"

RUN pip install xgboost catboost ipywidgets

RUN jupyter nbextension enable --py widgetsnbextension

RUN pip install lightgbm impyla gssapi thrift-sasl snappy python-snappy pyshark scapy nfstream pyorc datatable pymongo requests_ntlm
RUN pip install -U imbalanced-learn

RUN pip install -U scikit-image

#explainability
RUN pip install lime shap seaborn fastavro

#visualization
RUN pip install bokeh altair vega_datasets

# hyperparameter optimizations
RUN pip install -U "ray[tune]"
RUN pip install -U "ray[rllib]"
RUN pip install -U "ray[serve]"
RUN pip install optuna hyperopt bayesian-optimization bayesian-optimization

# more torch packages

RUN pip install pytorch-lightning pytorch-forecasting torchensemble rtdl transformers

# install pyspark

RUN pip install pyspark
RUN pip install pyspark[sql]
RUN pip install pyspark[pandas_on_spark] plotly

# instal hfds-fuse

#RUN apt install -y libprotobuf-c-dev
#RUN apt install -y protobuf-c-compiler
#RUN apt install -y libfuse-dev
#RUN apt install -y uncrustify


# install tensorflow

RUN pip install --upgrade tensorflow

# install beam

RUN pip install loguru
RUN mkdir /workspace/beamds/
COPY . /workspace/beamds/
RUN python -m pip install --upgrade build
WORKDIR /workspace/beamds/
RUN python -m build
RUN pip install dist/*.whl --force-reinstall
WORKDIR /workspace

# install db clinets next compilation

RUN apt install -y default-jdk
RUN pip install mysql-connector-python redis neo4j neo4j-driver splunk-sdk pysolr JPype1 JayDeBeApi PyHive
RUN pip install pymongo[gssapi,aws,ocsp,snappy,srv,tls,zstd,encryption]

# config jupyter notebook

RUN jupyter notebook --generate-config
RUN echo "c.NotebookApp.notebook_dir = '/home/'" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.allow_remote_access = True" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.token = ''" >> /root/.jupyter/jupyter_notebook_config.py

COPY beam_docker/entrypoint.sh /
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["10022", "10088", "12345678"]


