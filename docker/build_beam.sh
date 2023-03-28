#!/bin/bash

docker build -f docker/Dockerfile --tag beam:$(date '+%Y%m%d') . | tee -a /tmp/beam_build_$(date '+%Y%m%d').log