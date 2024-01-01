#!/bin/bash

INITIALS=$1
OPTIONAL_COMMAND=$2
MORE_ARGS=${@:3}

INITIALS=$(printf '%03d' $(echo $INITIALS | rev) | rev)

SSH_PORT="${INITIALS}22"
JUPYTER_PORT="${INITIALS}88"
MLFLOW_PORT="${INITIALS}80"
REDIS_PORT="${INITIALS}79"
RABBITMQ_PORT="${INITIALS}72"
PREFECT_PORT="${INITIALS}20"
RAY_REDIS_PORT="${INITIALS}78"
RAY_DASHBOARD_PORT="${INITIALS}90"
MONGODB_PORT="${INITIALS}17"

ROOT_PASSWORD="12345678"

echo "SSH Port: $SSH_PORT"
echo "Jupyter Port: $JUPYTER_PORT"
echo "MLflow Port: $MLFLOW_PORT"
echo "Redis Port: $REDIS_PORT"
echo "RabbitMQ Port: $RABBITMQ_PORT"
echo "Prefect Port: $PREFECT_PORT"
echo "Ray Redis Port: $RAY_REDIS_PORT"
echo "Ray Dashboard Port: $RAY_DASHBOARD_PORT"
echo "MongoDB Port: $MONGODB_PORT"
echo "Root password was updated"

export SSH_PORT=$SSH_PORT
export JUPYTER_PORT=$JUPYTER_PORT
export MLFLOW_PORT=$MLFLOW_PORT
export REDIS_PORT=$REDIS_PORT
export RABBITMQ_PORT=$RABBITMQ_PORT
export PREFECT_PORT=$PREFECT_PORT
export RAY_REDIS_PORT=$RAY_REDIS_PORT
export RAY_DASHBOARD_PORT=$RAY_DASHBOARD_PORT
export MONGODB_PORT=$MONGODB_PORT

echo "Port $SSH_PORT" >>/etc/ssh/sshd_config
echo "root:$ROOT_PASSWORD" | chpasswd

# Update Redis configuration
REDIS_CONF="/etc/redis/redis.conf"
sed -i "s/^port .*/port $REDIS_PORT/" $REDIS_CONF
sed -i 's/bind 127.0.0.1/bind 0.0.0.0/' $REDIS_CONF

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
echo "prefect_port, ${PREFECT_PORT}" >> /workspace/configuration/config.csv
echo "ray_redis_port, ${RAY_REDIS_PORT}" >> /workspace/configuration/config.csv
echo "ray_dashboard_port, ${RAY_DASHBOARD_PORT}" >> /workspace/configuration/config.csv
echo "mongodb_port, ${MONGODB_PORT}" >> /workspace/configuration/config.csv

bash /workspace/install/setup_env_vars.sh

jupyter-lab &
service ssh start
service redis-server start
service rabbitmq-server start
# run mlflow serve
mlflow server --host 0.0.0.0 --port "$MLFLOW_PORT" --backend-store-uri /workspace/mlruns --default-artifact-root /workspace/mlruns --workers 1 &
export MLFLOW_TRACKING_URI=http://localhost:$MLFLOW_PORT

# run prefect server

prefect server start --host 0.0.0.0 --port $PREFECT_PORT &

# run ray serve
ray start --head --node-ip-address=0.0.0.0 --port=${RAY_REDIS_PORT} --dashboard-port=${RAY_DASHBOARD_PORT} --dashboard-host=0.0.0.0 &


# run mongodb
bash /workspace/install/run_mongo.sh $MONGODB_PORT

#update-alternatives --set iptables /usr/sbin/iptables-legacy
#update-alternatives --set ip6tables /usr/sbin/ip6tables-legacy
#
## redirecting ports
#iptables -t nat -A PREROUTING -p tcp --dport 5672 -j REDIRECT --to-port $RABBITMQ_PORT
#iptables -t nat -A PREROUTING -p tcp --dport 6379 -j REDIRECT --to-port $REDIS_PORT
#iptables -t nat -A PREROUTING -p tcp --dport 5000 -j REDIRECT --to-port $MLFLOW_PORT
#iptables -t nat -A PREROUTING -p tcp --dport 27017 -j REDIRECT --to-port $MONGODB_PORT



if [ -z "$OPTIONAL_COMMAND" ]; then
    # If OPTIONAL_COMMAND is empty, run bash
    bash
else
    echo /etc/motd
    echo "Running command: ${OPTIONAL_COMMAND} ${MORE_ARGS}"
    # If OPTIONAL_COMMAND is provided, run it
    eval "${OPTIONAL_COMMAND} ${MORE_ARGS}"
fi
