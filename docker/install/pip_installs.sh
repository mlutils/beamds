#!/bin/bash

pip install --upgrade pip

# install pytorch-geometric
pip install torch-scatter torch-sparse torch-geometric torch-spline-conv torch-cluster torch-geometric-temporal
pip install xgboost catboost lightgbm


# additional python packages

pip install elasticsearch fasttext pandarallel gensim umap-learn pyvis graphviz pygraphviz opencv-contrib-python-headless
pip install dask[complete]

pip install loguru sphinx-server gunicorn xeus-python voila thop setuptools seldon-core build

pip install impyla gssapi thrift-sasl python-snappy pyshark scapy nfstream pyorc datatable
pip install requests_ntlm thriftpy fastparquet openpyxl mlflow pykeen tabulate onnx onnxruntime

# RUN pip install sasl

pip install -U zoopt scikit-optimize nevergrad HEBO flaml flaml[blendsearch]
pip install -U imbalanced-learn scapy scikit-image scikit-plot configurable-http-proxy

#explainability and visualization
pip install lime shap seaborn plotly fastavro bokeh altair vega_datasets dython fastai PyQt5 cdlib stellargraph twine
pip install nbtop botorch cma six Pillow pytorch-tabnet captum thop

# hyperparameter optimizations
pip install -U ray[tune,rllib,serve]
pip install optuna hyperopt bayesian-optimization deepchecks

# more torch packages

pip install pytorch-lightning pytorch-forecasting torchensemble rtdl transformers

# install pyspark
pip install pyspark pyspark[sql] pyspark[pandas_on_spark]

# install tensorflow
pip install -U tensorflow

# install jupyter
pip install -U jupyterlab ipympl notebook jupyterhub jupyter-resource-usage ipywidgets

# install db clinets next compilation

pip install mysql-connector-python redis neo4j neo4j-driver splunk-sdk pysolr JPype1 JayDeBeApi PyHive
pip install pymongo[gssapi,aws,ocsp,snappy,srv,tls,zstd,encryption]


pip install -U scikit-learn -U pandas -U networkx