#!/bin/bash

# Specify the environment variables you want to include
declare -a vars=("NVIDIA_VISIBLE_DEVICES" "NVIDIA_REQUIRE_CUDA" "CUDA_VERSION"
                 "LD_LIBRARY_PATH" "PATH" "CUDA_HOME" "CUDA_DRIVER_VERSION"
                 "OMPI_MCA_coll_hcoll_enable" "OPAL_PREFIX")

# Loop through each environment variable in the array
for var in "${vars[@]}"; do
    # Print the variable name
    echo "Variable name: $var"
    # Print the variable value
    echo "Variable value: ${!var}"
done

# List of users
users=("root" "$USER_NAME")

# Loop through each user
for user in "${users[@]}"; do
    if [ "$user" == "root" ]; then
        # Handle the root user specifically
        ssh_dir="/root/.ssh"
        echo "" > "$ssh_dir/environment"
    else
        # Handle other users
        ssh_dir="$USER_HOME_DIR/.ssh"
    fi

    # Create the .ssh directory for the user if it doesn't exist
    mkdir -p "$ssh_dir"

    # Create or clear the existing .ssh/environment file for each user
    echo "" > "$ssh_dir/environment"

    # Loop through each environment variable
    for var in "${vars[@]}"; do
        if [[ -v "$var" ]]; then
            # Append the environment variable to the user's .ssh/environment file
            echo "$var=${!var}" >> "$ssh_dir/environment"
        fi
    done

    # Ensure correct permissions are set for each user's .ssh directory and environment file
    chmod 700 "$ssh_dir"
    chmod 600 "$ssh_dir/environment"

    if [ "$user" == "root" ]; then
        chown -R root:root "$ssh_dir"
    else
        chown -R "$user:$user" "$ssh_dir"
    fi
done

# Loop through and copy the specified environment variables to root's environment file
for var in "${vars[@]}"; do
    if [[ -v "$var" ]]; then
        echo "$var=${!var}" >> /root/.ssh/environment
    fi
done

# Ensure correct permissions are set for root's .ssh environment file
chmod 600 /root/.ssh/environment
