#!/bin/bash

export PREFIX=/usr/local
export CUSIGNAL_HOME=/workspace/cusignal
git clone https://github.com/rapidsai/cusignal.git $CUSIGNAL_HOME
cd $CUSIGNAL_HOME
./build.sh
cd /workspace
