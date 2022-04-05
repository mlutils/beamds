#!/bin/sh

SSH_PORT=$1
JUPYTER_PORT=$2
ROOT_PASSWORD=$3

echo "

██████╗░███████╗░█████╗░███╗░░░███╗░░░░░░██████╗░░██████╗
██╔══██╗██╔════╝██╔══██╗████╗░████║░░░░░░██╔══██╗██╔════╝
██████╦╝█████╗░░███████║██╔████╔██║█████╗██║░░██║╚█████╗░
██╔══██╗██╔══╝░░██╔══██║██║╚██╔╝██║╚════╝██║░░██║░╚═══██╗
██████╦╝███████╗██║░░██║██║░╚═╝░██║░░░░░░██████╔╝██████╔╝
╚═════╝░╚══════╝╚═╝░░╚═╝╚═╝░░░░░╚═╝░░░░░░╚═════╝░╚═════╝░"


echo "SSH Port: $SSH_PORT"
echo "Jupyter Port: $JUPYTER_PORT"
echo "Root password was updated"

echo "Port $SSH_PORT" >> /etc/ssh/sshd_config
export JUPYTER_PORT=$JUPYTER_PORT
echo "root:$ROOT_PASSWORD" | chpasswd

service ssh start
jupyter notebook &

bash