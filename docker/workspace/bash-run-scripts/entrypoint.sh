#!/bin/bash

echo "Starting entrypoint script..."
# shellcheck disable=SC2145
echo "Initial arguments: $@"

# Check if the first argument is a path to a shell script
if [[ -f "$1" ]] && [[ $1 == *.sh ]]; then
    echo "Detected custom shell script as the first argument. Executing it instead of the default Beam entrypoint..."
    chmod +x "$1"
    "$@"
    exit
fi

## this manipulates the initials to 10 if no argument is passed - initial is passed in docker run command as -e INITIALS=$INITIALS for now
#INITIALS=$1
#shift

echo "Initials set to: $INITIALS"

# Initialize service flags
RUN_MLFLOW=true
RUN_JUPYTER=true
RUN_SSH=true
RUN_REDIS=true
RUN_RABBITMQ=true
RUN_PREFECT=false
RUN_RAY=true
RUN_MONGO=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-mlflow) RUN_MLFLOW=false; shift ;;
        --no-mongo) RUN_MONGO=false; shift ;;
        --mongo) RUN_MONGO=true; shift ;;
        --no-jupyter) RUN_JUPYTER=false; shift ;;
        --no-ssh) RUN_SSH=false; shift ;;
        --no-redis) RUN_REDIS=false; shift ;;
        --no-rabbitmq) RUN_RABBITMQ=false; shift ;;
        --no-prefect) RUN_PREFECT=false; shift ;;
        --no-ray) RUN_RAY=false; shift ;;
        --mlflow) RUN_MLFLOW=true; shift ;;
        --jupyter) RUN_JUPYTER=true; shift ;;
        --ssh) RUN_SSH=true; shift ;;
        --redis) RUN_REDIS=true; shift ;;
        --rabbitmq) RUN_RABBITMQ=true; shift ;;
        --prefect) RUN_PREFECT=true; shift ;;
        --ray) RUN_RAY=true; shift ;;
        *) break ;;
    esac
done

echo "Root password was updated"
ROOT_PASSWORD="qwerqwer"
echo "Beam password was updated"
BEAM_PASSWORD="12345678"

OPTIONAL_COMMAND=$2
MORE_ARGS=${@:3}

# Debug: Print INITIALS before and after formatting
echo "INITIALS before formatting: $INITIALS"
#INITIALS=$(printf '%03d' $(echo $INITIALS | rev) | rev)
echo "Formatted INITIALS: ${INITIALS}"

# Set environment variables
bash /workspace/bash-run-scripts/setup_env_vars.sh

# add the initials and the image name to the motd

touch /workspace/configuration/config.csv

echo "parameters, value" >> /workspace/configuration/config.csv
echo "initials, ${INITIALS}" >> /workspace/configuration/config.csv

if [ "$RUN_REDIS" = true ]; then
  REDIS_PORT="${INITIALS}79"
  echo "Redis Port: $REDIS_PORT"
  export REDIS_PORT=$REDIS_PORT
  echo "redis_port, ${REDIS_PORT}" >> /workspace/configuration/config.csv
  REDIS_CONF="/etc/redis/redis.conf"
  sed -i "s/^port .*/port $REDIS_PORT/" $REDIS_CONF
  sed -i 's/bind 127.0.0.1/bind 0.0.0.0/' $REDIS_CONF
  service redis-server start
  echo "Redis is running."
else
  echo "Redis is disabled."
fi

if [ "$RUN_RABBITMQ" = true ]; then
  RABBITMQ_PORT="${INITIALS}72"
  echo "RabbitMQ Port: $RABBITMQ_PORT"
  export RABBITMQ_PORT=$RABBITMQ_PORT
  echo "rabbitmq_port, ${RABBITMQ_PORT}" >> /workspace/configuration/config.csv
  RABBITMQ_CONF="/etc/rabbitmq/rabbitmq.conf"
  if [ ! -f "$RABBITMQ_CONF" ]; then
      echo "listeners.tcp.default = $RABBITMQ_PORT" > $RABBITMQ_CONF
  else
      sed -i "s/listeners.tcp.default = .*/listeners.tcp.default = $RABBITMQ_PORT/" $RABBITMQ_CONF
  fi
  service rabbitmq-server start &
  echo "RabbitMQ is running."
else
  echo "RabbitMQ is disabled."
fi

if [ "$RUN_MLFLOW" = true ]; then
  MLFLOW_PORT="${INITIALS}80"
  echo "MLflow Port: $MLFLOW_PORT"
  export MLFLOW_PORT=$MLFLOW_PORT
  echo "mlflow_port, ${MLFLOW_PORT}" >> /workspace/configuration/config.csv
  which mlflow
  if [ $? -ne 0 ]; then
    echo "MLflow command not found, installing MLflow..."
    pip install mlflow
  fi
  mlflow server --host 0.0.0.0 --port "$MLFLOW_PORT" --backend-store-uri /workspace/mlruns --default-artifact-root /workspace/mlruns --workers 1 &
  export MLFLOW_TRACKING_URI=http://localhost:$MLFLOW_PORT
  echo "MLflow server is running."
else
  echo "MLflow is disabled."
fi

if [ "$RUN_PREFECT" = true ]; then
  PREFECT_PORT="${INITIALS}20"
  echo "Prefect Port: $PREFECT_PORT"
  export PREFECT_PORT=$PREFECT_PORT
  echo "prefect_port, ${PREFECT_PORT}" >> /workspace/configuration/config.csv
  prefect server start --host 0.0.0.0 --port $PREFECT_PORT &
  echo "Prefect server is running."
else
  echo "Prefect is disabled."
fi

if [ "$RUN_MONGO" = true ]; then
  MONGODB_PORT="${INITIALS}17"
  echo "MongoDB Port: $MONGODB_PORT"
  export MONGODB_PORT=$MONGODB_PORT
  echo "mongodb_port, ${MONGODB_PORT}" >> /workspace/configuration/config.csv
  bash /workspace/bash-run-scripts/run_mongo.sh $MONGODB_PORT
  echo "MongoDB server is running."
else
  echo "MongoDB is disabled."
fi

if [ "$RUN_RAY" = true ]; then
  RAY_REDIS_PORT="${INITIALS}78"
  RAY_DASHBOARD_PORT="${INITIALS}90"
  echo "Ray Redis Port: $RAY_REDIS_PORT"
  echo "Ray Dashboard Port: $RAY_DASHBOARD_PORT"
  export RAY_REDIS_PORT=$RAY_REDIS_PORT
  export RAY_DASHBOARD_PORT=$RAY_DASHBOARD_PORT
  echo "ray_redis_port, ${RAY_REDIS_PORT}" >> /workspace/configuration/config.csv
  echo "ray_dashboard_port, ${RAY_DASHBOARD_PORT}" >> /workspace/configuration/config.csv
  ray start --head --node-ip-address=0.0.0.0 --port=${RAY_REDIS_PORT} --dashboard-port=${RAY_DASHBOARD_PORT} --dashboard-host=0.0.0.0 &
  echo "Ray server is running."
else
  echo "Ray is disabled."
fi

if [ "$RUN_SSH" = true ]; then
  SSH_PORT="${INITIALS}22"
  echo "SSH Port: $SSH_PORT"
  export SSH_PORT=$SSH_PORT
  echo "ssh_port, ${SSH_PORT}" >> /workspace/configuration/config.csv
  cp /workspace/configuration/sshd_config /opt/ssh/sshd_config
  echo "Port $SSH_PORT" >>/opt/ssh/sshd_config
  echo "root:$ROOT_PASSWORD" | chpasswd
  echo "beam:$BEAM_PASSWORD" | chpasswd
  supervisord -c /etc/supervisor/supervisord.conf &> /tmp/supervisor.log &
  echo "SSH is running."
else
  echo "SSH is disabled."
fi

if [ "$RUN_JUPYTER" = true ]; then
  JUPYTER_PORT="${INITIALS}88"
  echo "Jupyter Port: $JUPYTER_PORT"
  export JUPYTER_PORT=$JUPYTER_PORT
  echo "jupyter_port, ${JUPYTER_PORT}" >> /workspace/configuration/config.csv
  su - beam -c "jupyter-lab --port=$JUPYTER_PORT" &
  echo "Jupyter is running."
else
  echo "Jupyter is disabled."
fi

service start avahi-daemon
service enable avahi-daemon

if [ -z "$OPTIONAL_COMMAND" ]; then
    bash
else
    echo /etc/motd
    echo "Running command: ${OPTIONAL_COMMAND} ${MORE_ARGS}"
    eval "${OPTIONAL_COMMAND} ${MORE_ARGS}"
fi
echo "Entrypoint script completed."