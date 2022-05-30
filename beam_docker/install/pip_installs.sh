#!/bin/sh


# additional python packages

pip install elasticsearch fasttext pandarallel gensim umap-learn pyvis graphviz pygraphviz opencv-python
pip install "dask[complete]"

pip install xgboost catboost ipywidgets loguru sphinx-server gunicorn xeus-python voila

pip install lightgbm impyla gssapi thrift-sasl snappy python-snappy pyshark scapy nfstream pyorc datatable pymongo
pip install requests_ntlm thriftpy fastparquet openpyxl

# RUN pip install sasl

pip install -U imbalanced-learn scapy scikit-image scikit-plot jupyterhub configurable-http-proxy

#explainability and visualization
pip install lime shap seaborn fastavro bokeh altair vega_datasets dython fastai PyQt5 cdlib
pip install nbtop

# hyperparameter optimizations
pip install -U "ray[tune]"
pip install -U "ray[rllib]"
pip install -U "ray[serve]"
pip install optuna hyperopt bayesian-optimization bayesian-optimization deepchecks

# more torch packages

pip install pytorch-lightning pytorch-forecasting torchensemble rtdl transformers

# install pyspark

pip install pyspark
pip install pyspark[sql]
pip install pyspark[pandas_on_spark] plotly


# install tensorflow
pip install -U jupyterlab tensorflow ipympl notebook
pip install jupyter-resource-usage

# install db clinets next compilation

pip install mysql-connector-python redis neo4j neo4j-driver splunk-sdk pysolr JPype1 JayDeBeApi PyHive
pip install pymongo[gssapi,aws,ocsp,snappy,srv,tls,zstd,encryption]