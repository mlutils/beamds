# BeamDS (Beam Data Science)

<p align="center">
<img src="resources/beam_icon.png" width="200">
</p>

## What is Beam for?

Beam was created by data-science practitioners for data-science practitioners. It is designed as an ecosystem for developing and deploying data-driven algorithms in Python. It aims to increase productivity, efficiency, and performance in the research phase and to provide production-grade tools in the deployment part.

## Our Guiding Principles

1. Support all phases of data-driven algorithm development:
    1. Data exploration
    2. Data manipulation, preprocessing, and ETLs (Extract, Transform and Load)
    3. Algorithm selection
    4. Algorithm training
    5. Hyperparameter tuning
    6. odel deployment
    7. Lifelong learning
2. Production level coding from the first line of code: no more quick and dirty Proof Of Concepts (POC). Every line of code counts toward a production model.
3. Consume effectively all resources: use multi-core, multi-GPUs, distributed computing, remote storage solutions, and databases to enable as much as possible productivity by the resources at hand.
4. Be agile: Development and production environments can change rapidly. Beam minimizes the friction of changing environments, filesystems, and computing resources to almost zero.
5. Be efficient: every line of code in Beam is optimized to be as efficient as possible and to avoid unnecessary overheads.
6. Easy to deploy and use algorithms: make deployment as easy as a line of code, import remote algorithms and services by their URI, and no more.
7. Excel your algorithms: Beam comes with some state-of-the-art deep neural network implementations. Beam will help you store, analyze, and return to your running experiments with ease. When you are done, with development, beam will help you optimize your hyperparameters on your GPU machines.
8. Data can be a hassle: beam can manipulate complex and nested data structures, including reading, processing, chunking, multi-processing, error handling, and writing.
9. Be relevant: Beam is committed to staying relevant and updating towards the future of AI, adding support for Large Language Models (LLMs) and more advanced algorithms.
10. Beam is the Swiss army knife that gets into your pocket: it is easy to install and maintain and it comes with the Beam Docker Image s.t. you can start developing and creating with zero effort even without an internet connection.


## Installation

To install the full package from PyPi use:
```shell
pip install beam-ds[all]
```
If you want to install only the data-science related components use:
```shell
pip install beam-ds[ds]
``` 
To install only the LLM (Large Language Model) related components use:
```shell
pip install beam-ds[llm]
```

The prerequisite packages will be installed automatically, they can be found in the setup.cfg file.

## Build from source

This BeamDS implementation follows the guide at 
https://packaging.python.org/tutorials/packaging-projects/

install the build package:
```shell
python -m pip install --upgrade build
```

to reinstall the package after updates use:

1. Now run this command from the same directory where pyproject.toml is located:
```shell
python -m build
```
   
2. reinstall the package with pip:
```shell
pip install dist/*.whl --force-reinstall
```

## Getting Started

There are several examples both in .py files (in the examples folder) and in jupyter notebooks (in the notebooks folder).
Specifically, you can start by looking into the beam_resources.ipynb notebook which makes you familiar the different
resources available in Beam.

[Go To the beam_resource.ipynb page](/notebooks/beam_resource.ipynb)

## Building the Beam-DS docker image

The docker image is based on the latest official NVIDIA pytorch image.
To build the docker image from Ubuntu host, you need to:

1. update nvidia drivers to the latest version:
https://linuxconfig.org/how-to-install-the-nvidia-drivers-on-ubuntu-20-04-focal-fossa-linux

2. install docker:
https://docs.docker.com/desktop/linux/install/ubuntu/

3. Install NVIDIA container toolkit:
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#install-guide

4. Install and configure NVIDIA container runtime:
https://stackoverflow.com/a/61737404

## Build the sphinx documentation

Follow https://github.com/cimarieta/sphinx-autodoc-example

## Profiling your code with Scalene

Scalene is a high-performance python profiler that supports GPU profiling. 
To analyze your code with Scalene use the following arguments:
```shell
scalene --reduced-profile --outfile OUTFILE.html --html --- your_prog.py <your additional arguments>
```

## Uploading the package to PyPi

1. Install twine:
```shell
python -m pip install --user --upgrade twine
```

2. Build the package:
```shell
python -m build
```

3. Upload the package:
```shell
python -m twine upload --repository pypi dist/* 
```








