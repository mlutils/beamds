#!/bin/bash

INITIALS=$1
INITIALS=$(printf '%03d' $(echo $INITIALS | rev) | rev)

SSH_PORT="${INITIALS}22"
JUPYTER_PORT="${INITIALS}88"
ROOT_PASSWORD="12345678"

echo "SSH Port: $SSH_PORT"
echo "Jupyter Port: $JUPYTER_PORT"
echo "Root password was updated"

echo "Port $SSH_PORT" >>/etc/ssh/sshd_config
export JUPYTER_PORT=$JUPYTER_PORT
echo "root:$ROOT_PASSWORD" | chpasswd

# write the beam configuration to a file

mkdir /workspace/configuration
touch /workspace/configuration/config.csv

echo "parameters, value" >> /workspace/configuration/config.csv
echo "initials, ${INITIALS}" >> /workspace/configuration/config.csv
echo "ssh_port, ${SSH_PORT}" >> /workspace/configuration/config.csv
echo "jupyter_port, ${JUPYTER_PORT}" >> /workspace/configuration/config.csv

service ssh start
jupyter-lab &

bash
