# BeamDS Package (beam data-science)

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








