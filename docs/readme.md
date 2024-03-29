# BeamDS Package (beam data-science)

<p align="center">
<img src="https://user-images.githubusercontent.com/32983309/175893461-19eeaacb-ddf0-43fd-b43c-20a9144ac65d.png" width="200">
</p>

This BeamDS implementation follows the guide at 
https://packaging.python.org/tutorials/packaging-projects/

prerequisits:

install the build package:
```shell
python -m pip install --upgrade build
```

Packages to install:
```
tqdm, loguru, tensorboard
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








