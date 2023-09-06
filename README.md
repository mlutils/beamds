# BeamDS Package (beam data-science)

<p align="center">
<img src="https://user-images.githubusercontent.com/32983309/175893461-19eeaacb-ddf0-43fd-b43c-20a9144ac65d.png" width="200">
</p>

This BeamDS implementation follows the guide at 
https://packaging.python.org/tutorials/packaging-projects/

prerequisits:

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








