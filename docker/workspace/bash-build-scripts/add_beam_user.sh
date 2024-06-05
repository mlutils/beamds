# Create a non-root user
useradd -m -d "$USER_HOME_DIR" -s /bin/bash "$USER_NAME"
echo "$USER_NAME:12345678" | chpasswd

# change beam home directory to /USER_NAME
usermod -aG sudo "$USER_NAME"

echo "cat /etc/motd" >> "$USER_HOME_DIR/.bashrc"
echo "cd /home" >> ~/.bashrc

git config --global http.sslVerify false
