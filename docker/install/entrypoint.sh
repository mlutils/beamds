#!/bin/bash

INITIALS=$1
INITIALS=$(printf '%03d' $(echo $INITIALS | rev) | rev)

SSH_PORT="${INITIALS}22"
JUPYTER_PORT="${INITIALS}88"
MLFLOW_PORT="${INITIALS}80"
REDIS_PORT="${INITIALS}79"
RABBITMQ_PORT="${INITIALS}72"
PREFECT_PORT="${INITIALS}20"
ROOT_PASSWORD="12345678"

echo "SSH Port: $SSH_PORT"
echo "Jupyter Port: $JUPYTER_PORT"
echo "MLflow Port: $MLFLOW_PORT"
echo "Redis Port: $REDIS_PORT"
echo "RabbitMQ Port: $RABBITMQ_PORT"
echo "Root password was updated"

echo "Port $SSH_PORT" >>/etc/ssh/sshd_config
export JUPYTER_PORT=$JUPYTER_PORT
echo "root:$ROOT_PASSWORD" | chpasswd

# Update Redis configuration
REDIS_CONF="/etc/redis/redis.conf"
sed -i "s/^port .*/port $REDIS_PORT/" $REDIS_CONF

# Update RabbitMQ configuration
RABBITMQ_CONF="/etc/rabbitmq/rabbitmq.conf"
if [ ! -f "$RABBITMQ_CONF" ]; then
    echo "listeners.tcp.default = $RABBITMQ_PORT" > $RABBITMQ_CONF
else
    sed -i "s/listeners.tcp.default = .*/listeners.tcp.default = $RABBITMQ_PORT/" $RABBITMQ_CONF
fi

# Set Hadoop CLASSPATH
CLASSPATH=$(/usr/local/hadoop/bin/hadoop classpath --glob)

# write the beam configuration to a file

mkdir /workspace/configuration
touch /workspace/configuration/config.csv

echo "parameters, value" >> /workspace/configuration/config.csv
echo "initials, ${INITIALS}" >> /workspace/configuration/config.csv
echo "ssh_port, ${SSH_PORT}" >> /workspace/configuration/config.csv
echo "jupyter_port, ${JUPYTER_PORT}" >> /workspace/configuration/config.csv
echo "mlflow_port, ${MLFLOW_PORT}" >> /workspace/configuration/config.csv
echo "redis_port, ${REDIS_PORT}" >> /workspace/configuration/config.csv
echo "rabbitmq_port, ${RABBITMQ_PORT}" >> /workspace/configuration/config.csv

jupyter-lab &
service ssh start
service redis-server start
service rabbitmq-server start
# run mlflow serve
mlflow server --host 0.0.0.0 --port "$MLFLOW_PORT" --backend-store-uri /workspace/mlruns --default-artifact-root /workspace/mlruns --workers 1 &
export MLFLOW_TRACKING_URI=http://localhost:$MLFLOW_PORT

# run prefect server

prefect server start --host 0.0.0.0 --port $PREFECT_PORT

bash
