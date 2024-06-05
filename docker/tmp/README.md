use this directory to update installation files during the build process, when build failures occur.
copy here the files that are needed to update the installation files, fix it and then in the Dockerfile copy it back to the original location.
For example:
```Dockerfile
```
COPY docker/tmp/configure_jupyter.sh /workspace/bash-build-scripts/configure_jupyter.sh
RUN bash /workspace/bash-build-scripts/configure_jupyter.sh
```