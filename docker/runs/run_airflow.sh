WEBSERVER_PORT=$1

source airflow/bin/activate
airflow webserver --port "$WEBSERVER_PORT" &
airflow scheduler &
deactivate

# follow:
# https://chat.openai.com/share/a59ecc08-dcd2-470e-8e5e-ee1cf22a5dd7