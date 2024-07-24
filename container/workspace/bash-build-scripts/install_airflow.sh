
python -m venv airflow
source airflow/bin/activate
pip install --upgrade pip

export AIRFLOW_HOME=~/airflow
AIRFLOW_VERSION=2.8.2
PYTHON_VERSION="$(python --version | cut -d " " -f 2 | cut -d "." -f 1-2)"
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"


airflow users create \
    --username "$USER_NAME" \
    --firstname "$USER_NAME" \
    --lastname "$USER_NAME" \
    --role Admin \
    --email "$USER_NAME@example.com" \
    --password 12345678


deactivate