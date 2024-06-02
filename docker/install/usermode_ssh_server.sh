# Create a non-root user
useradd -m -s /bin/bash beam
echo 'beam:12345678' | chpasswd
usermod -aG sudo username
mkdir -p  /opt/ssh
mkdir -p  /opt/supervisor
cp /workspace/beam_image/supervisord.conf /etc/supervisor/conf.d/
cp /workspace/beam_image/sshd_config /opt/ssh
# for now this line throw unclear error when running in the Dockerfile, but it runs fine from the container
# so for now, we will run it from the container and commit the container to a modified image.
# RUN pip install git+https://github.com/chaoleili/jupyterlab_tensorboard.git
ssh-keygen -q -N "" -t dsa -f /opt/ssh/ssh_host_dsa_key
ssh-keygen -q -N "" -t rsa -b 4096 -f /opt/ssh/ssh_host_rsa_key
ssh-keygen -q -N "" -t ecdsa -f /opt/ssh/ssh_host_ecdsa_key
ssh-keygen -q -N "" -t ed25519 -f /opt/ssh/ssh_host_ed25519_key
# Configure SSH for non-root public key authentication
mkdir -p /home/beam/.ssh
chmod 700 /home/beam/.ssh
chown beam:beam /home/beam/.ssh
chown -R beam. /opt/ssh
#&& chmod 600 /home/beam/.ssh/authorized_keys && \
#    chown beam:beam /home/beam/.ssh/authorized_keys
