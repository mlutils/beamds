#!/bin/bash
SUFFIX=$1
MORE_ARGS=${@:2}

TAG=$(date '+%Y%m%d')${SUFFIX}

echo "Building image with tag: ${TAG}"

docker build -f docker/Dockerfile --tag beam:${TAG} --progress=plain "$MORE_ARGS" . 2>&1 | tee -a /tmp/beam_build_$(date '+%Y%m%d').log