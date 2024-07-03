
mkdir -p  /opt/ssh_user
mkdir -p  /opt/supervisor
mkdir -p  /run/sshd

cp /workspace/configuration/supervisord_user.conf /etc/supervisor/conf.d/
cp /workspace/configuration/sshd_config_root /etc/ssh/sshd_config
cp /workspace/configuration/sshd_config_user /opt/ssh_user/sshd_config

# for now this line throw unclear error when running in the dockerfile-beam, but it runs fine from the container
# so for now, we will run it from the container and commit the container to a modified image.
# RUN pip install git+https://github.com/chaoleili/jupyterlab_tensorboard.git

# Keys for SSH_USER
ssh-keygen -q -N "" -t dsa -f /opt/ssh_user/ssh_host_dsa_key
ssh-keygen -q -N "" -t rsa -b 4096 -f /opt/ssh_user/ssh_host_rsa_key
ssh-keygen -q -N "" -t ecdsa -f /opt/ssh_user/ssh_host_ecdsa_key
ssh-keygen -q -N "" -t ed25519 -f /opt/ssh_user/ssh_host_ed25519_key
# Configure SSH for non-root public key authentication

mkdir -p "$USER_HOME_DIR/.ssh"
chmod 700 "$USER_HOME_DIR/.ssh"
chown "$USER_NAME":"$USER_NAME" "$USER_HOME_DIR/.ssh"
chown -R "$USER_NAME". /opt/ssh_user

## replace string USER_NAME in /etc/supervisor/conf.d/ with $USER_NAME
#sed -i "s/USER_NAME/$USER_NAME/g" /etc/supervisor/conf.d/supervisord_user.conf
# replace string USER_NAME in /etc/supervisor/conf.d/ with $USER_NAME
sed -i "s/USER_NAME/$USER_NAME/g" /etc/supervisor/conf.d/supervisord_user.conf

#&& chmod 600 /home/beam/.ssh/authorized_keys && \
#  chown beam:beam /home/beam/.ssh/authorized_keys
