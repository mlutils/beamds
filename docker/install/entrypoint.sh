#!/bin/bash

INITIALS=$1

SSH_PORT="${INITIALS}022"
JUPYTER_PORT="${INITIALS}088"
ROOT_PASSWORD="${INITIALS}123456"


echo "SSH Port: $SSH_PORT"
echo "Jupyter Port: $JUPYTER_PORT"
echo "Root password was updated"

echo "Port $SSH_PORT" >> /etc/ssh/sshd_config
export JUPYTER_PORT=$JUPYTER_PORT
echo "root:$ROOT_PASSWORD" | chpasswd

service ssh start
jupyter-lab &

bash