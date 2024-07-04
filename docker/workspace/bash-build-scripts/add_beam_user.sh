# Check if the USER_NAME is 'root', skip further commands for 'root'
if [ "$USER_NAME" != "root" ]; then
    # Create a non-root user
    useradd -m -d "/home/$USER_NAME" -s /bin/bash "$USER_NAME"
    echo "$USER_NAME:12345678" | chpasswd

    # Add the new user to the sudo group
    usermod -aG sudo "$USER_NAME"

    # Update .bashrc for the new user to change directory upon login
    echo "cd /home/$USER_NAME" >> "/home/$USER_NAME/.bashrc"

    # Optionally set git to not verify SSL globally
    # Consider the security implications of this setting.
    git config --global http.sslVerify false
fi


#
## Create a non-root user
#useradd -m -d "$USER_HOME_DIR" -s /bin/bash "$USER_NAME"
#echo "$USER_NAME:12345678" | chpasswd
#
## change beam home directory to /USER_NAME
#usermod -aG sudo "$USER_NAME"
#
#echo "cd /home" >> ~/.bashrc
#git config --global http.sslVerify false
