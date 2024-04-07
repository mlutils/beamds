#!/bin/bash

# Specify the environment variables you want to include
declare -a vars=("NVIDIA_VISIBLE_DEVICES" "NVIDIA_REQUIRE_CUDA" "CUDA_VERSION"
                 "LD_LIBRARY_PATH" "PATH" "CUDA_HOME" "CUDA_DRIVER_VERSION"
                 "OMPI_MCA_coll_hcoll_enable" "OPAL_PREFIX")

# Create or clear the existing .ssh/environment file
echo "" > ~/.ssh/environment

# Loop through and copy the specified environment variables
for var in "${vars[@]}"; do
    if [[ -v "$var" ]]; then
        echo "$var=${!var}" >> ~/.ssh/environment
    fi
done

# Ensure correct permissions are set
chmod 600 ~/.ssh/environment