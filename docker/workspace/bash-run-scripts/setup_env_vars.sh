#!/bin/bash

# Specify the environment variables you want to include
declare -a vars=("NVIDIA_VISIBLE_DEVICES" "NVIDIA_REQUIRE_CUDA" "CUDA_VERSION"
                 "LD_LIBRARY_PATH" "PATH" "CUDA_HOME" "CUDA_DRIVER_VERSION"
                 "OMPI_MCA_coll_hcoll_enable" "OPAL_PREFIX")

# List of users
users=("root" "$USER_NAME")

for user in "${users[@]}"; do
    if [ "$user" == "root" ]; then
        # Handle the root user specifically
        ssh_dir="/root/.ssh"
        profile_file="/root/.profile"
    else
        # Handle other users
        ssh_dir="$USER_HOME_DIR/.ssh"
        profile_file="$USER_HOME_DIR/.profile"
    fi

    # Create the .ssh directory for the user if it doesn't exist
    mkdir -p "$ssh_dir"

    # File to store environment variable settings
    env_file="$ssh_dir/set_env.sh"
    echo "#!/bin/bash" > "$env_file"

    # Loop through each environment variable
    for var in "${vars[@]}"; do
        if [[ -v "$var" ]]; then
            # Append the environment variable export to the script file
            echo "export $var='${!var}'" >> "$env_file"
        fi
    done

    # Make the environment script executable
    chmod +x "$env_file"

    # Append source command to .profile or .bashrc to source the variables on login
    echo "source '$env_file'" >> "$profile_file"

    # Ensure correct permissions are set for each user's .ssh directory and profile file
    chmod 700 "$ssh_dir"
    chmod 600 "$profile_file"

    if [ "$user" == "root" ]; then
        chown -R root:root "$ssh_dir"
        chown root:root "$profile_file"
    else
        chown -R "$user:$user" "$ssh_dir"
        chown "$user:$user" "$profile_file"
    fi
done
