Auto generation of .rst files with sphinx-apidoc
```shell
sphinx-apidoc -o docs/source beamds -d 4
```

To build the html files on a local machine:

```shell
cd docs
sphinx-build -b html . ../../api
```