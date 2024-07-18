#!/bin/bash

# Print the current value of USER_NAME for debugging
echo "Current USER_NAME is: '$USER_NAME'"

# Check if the USER_NAME is 'root', skip further commands for 'root'
if [ "$USER_NAME" != "root" ]; then
    # Create a non-root user
    useradd -m -d "$USER_HOME_DIR" -s /bin/bash "$USER_NAME"
    if [ $? -eq 0 ]; then
        echo "User $USER_NAME added successfully."
    else
        echo "Failed to add user $USER_NAME."
        exit 1
    fi

    echo "$USER_NAME:12345678" | chpasswd

    # Add the new user to the sudo group
    usermod -aG sudo "$USER_NAME"

    # Update .bashrc for the new user to change directory upon login
    echo "cd $USER_HOME_DIR" >> "$USER_HOME_DIR/.bashrc"

    # Optionally set git to not verify SSL globally
    # Consider the security implications of this setting.
    sudo -u "$USER_NAME" git config --global http.sslVerify false
else
    echo "Skipping operations for root user."
fi
