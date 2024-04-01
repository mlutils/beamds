#!/bin/bash
#yszzss2211
# INITIALS=$1
# INITIALS=$(printf '%03d' $(echo $INITIALS | rev) | rev)

# SSH_PORT="${INITIALS}22"
# JUPYTER_PORT="${INITIALS}88"
# MLFLOW_PORT="${INITIALS}80"
# REDIS_PORT="${INITIALS}79"
# RABBITMQ_PORT="${INITIALS}72"
# ROOT_PASSWORD="12345678"

SSH_PORT="2222"
JUPYTER_PORT="8888"
MLFLOW_PORT="8880"
REDIS_PORT="7779"
RABBITMQ_PORT="7772"
ROOT_PASSWORD="12345678"

echo "SSH Port: $SSH_PORT"
echo "Jupyter Port: $JUPYTER_PORT"
echo "MLflow Port: $MLFLOW_PORT"
echo "Redis Port: $REDIS_PORT"
echo "RabbitMQ Port: $RABBITMQ_PORT"
echo "Root password was updated"
sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

initialization_task() {
    # Your initialization code here
    echo "Port $SSH_PORT" >>/opt/ssh/sshd_config
    # More initialization...
}
initialization_task &

export JUPYTER_PORT=$JUPYTER_PORT
mkdir /var/run/sshd
#echo "root:$ROOT_PASSWORD" | chpasswdd
#sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

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

pip install jupyterlab-tensorboard-pro notebook-shim async_lru

jupyter nbextension enable --py widgetsnbextension

jupyter-lab --generate-config
echo "c.ServerApp.notebook_dir = '/home/'" >> /root/.jupyter/jupyter_lab_config.py
echo "c.ServerApp.allow_remote_access = True" >> /root/.jupyter/jupyter_lab_config.py
echo "c.ServerApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_lab_config.py
echo "c.ServerApp.token = ''" >> /root/.jupyter/jupyter_lab_config.py
echo "c.ServerApp.allow_root = True" >> /root/.jupyter/jupyter_lab_config.py


echo "c.LabServerApp.open_browser = False" >> /root/.jupyter/jupyter_lab_config.py


jupyter-lab &
#service ssh start
service redis-server start
service rabbitmq-server start
# run mlflow serve
mlflow server --host 0.0.0.0 --port "$MLFLOW_PORT" --backend-store-uri /workspace/mlruns --default-artifact-root /workspace/mlruns --workers 1 &
export MLFLOW_TRACKING_URI=http://localhost:$MLFLOW_PORT
service ssh stop
#bash
#/usr/sbin/sshd -D
echo "Port $SSH_PORT" >>/opt/ssh/sshd_config

/usr/bin/supervisord -c /etc/supervisor/supervisord.conf &> /tmp/supervisor.log

sleep infinity
