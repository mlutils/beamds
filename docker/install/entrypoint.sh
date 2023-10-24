#!/bin/bash

INITIALS=$1
INITIALS=$(printf '%03d' $(echo $INITIALS | rev) | rev)

SSH_PORT="${INITIALS}22"
JUPYTER_PORT="${INITIALS}88"
MLFLOW_PORT="${INITIALS}80"
ROOT_PASSWORD="12345678"

echo "SSH Port: $SSH_PORT"
echo "Jupyter Port: $JUPYTER_PORT"
echo "Root password was updated"

echo "Port $SSH_PORT" >>/etc/ssh/sshd_config
export JUPYTER_PORT=$JUPYTER_PORT
echo "root:$ROOT_PASSWORD" | chpasswd

# run mlflow server
mlflow server --host 0.0.0.0 --port "$MLFLOW_PORT" --backend-store-uri /workspace/mlruns --default-artifact-root /workspace/mlruns --workers 1 &
export MLFLOW_TRACKING_URI=http://localhost:$MLFLOW_PORT

# Set Hadoop CLASSPATH
CLASSPATH=$(/usr/local/hadoop/bin/hadoop classpath --glob)

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
