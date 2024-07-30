#!/bin/bash
SUFFIX=$1
TAG=$(date '+%Y%m%d')${SUFFIX}

echo "Building image with tag: ${TAG}"

docker build -f container/dockerfile-beam --tag beam:${TAG} --progress=plain . 2>&1 | tee -a /tmp/beam_build_$(date '+%Y%m%d').log