from flask import Flask, render_template, url_for, send_file, request
import argparse
import os
import pandas as pd

app = Flask(__name__)


def get_route(name, base_url="http://localhost:"):

    k8s_name = f"KUBERNETES_{name.upper()}_ROUTE_HOST"
    if os.environ.get(k8s_name):
        return f"http://{os.environ.get(k8s_name)}"
    else:
        conf = pd.read_csv('/workspace/configuration/config.csv', index_col=0)
        port = int(conf.drop_duplicates().loc[f"{name.lower()}_port"])
        return f"{base_url}{port}"


@app.route('/')
def home():
    # Get the scheme (http or https)
    scheme = request.scheme

    # Get the host, including the port if specified
    host = request.host.split(':')[0]
    base_url = f"{scheme}://{host}:"

    services = [
        {"name": "MLflow", "icon": url_for('static', filename='mlflow_icon.png'),
         "url": get_route('mlflow', base_url=base_url),
         "description": "MLflow is an open-source platform for managing the end-to-end machine learning lifecycle."},
        {"name": "Jupyter", "icon": url_for('static', filename='jupyter_icon.png'),
         "url": get_route('jupyter', base_url=base_url),
         "description": "Jupyter Notebook is an open-source web application that allows you to create and share documents that contain live code, equations, visualizations, and narrative text."},
        {"name": "SSH", "icon": url_for('static', filename='ssh_icon.png'), "url": url_for('generate_ssh_script'),
         "description": "SSH (Secure Shell) is a protocol for securely accessing network services over an unsecured network."},
        {"name": "Ray", "icon": url_for('static', filename='ray_icon.png'),
         "url": get_route('ray_dashboard', base_url=base_url),
         "description": "Ray is a distributed computing framework for parallel and distributed Python."},
        {"name": "Chroma", "icon": url_for('static', filename='chroma_icon.png'), "url": "Coming Soon",
         "description": "Chroma is a vector database for building high-performance machine learning applications."},
        {"name": "Redis", "icon": url_for('static', filename='redis_icon.png'), "url": "Coming Soon",
         "description": "Redis is an in-memory data structure store used as a database, cache, and message broker."},
        {"name": "RabbitMQ", "icon": url_for('static', filename='rabbitmq_icon.png'), "url": "Coming Soon",
         "description": "RabbitMQ is an open-source message broker software that implements the Advanced Message Queuing Protocol (AMQP)."},
        {"name": "Prefect", "icon": url_for('static', filename='prefect_icon.png'), "url": "Coming Soon",
         "description": "Prefect is a workflow orchestration tool that helps you to coordinate and monitor the execution of data workflows."},
        {"name": "MongoDB", "icon": url_for('static', filename='mongo_icon.png'), "url": "Coming Soon",
         "description": "MongoDB is a source-available cross-platform document-oriented database program."},
        {"name": "MinIO", "icon": url_for('static', filename='minio_icon.png'), "url": "Coming Soon",
         "description": "MinIO is a high-performance, S3 compatible object storage system."},
    ]

    root_icon = url_for('static', filename='beam_docker_icon.webp')

    return render_template('index.html', services=services, root_icon=root_icon)


@app.route('/generate_ssh_script')
def generate_ssh_script():
    # Parameters for the SSH connection
    ssh_ip = "your_ssh_ip"
    ssh_port = "your_ssh_port"
    ssh_user = "your_ssh_username"

    # Script content
    script_content = f"""#!/bin/bash
ssh {ssh_user}@{ssh_ip} -p {ssh_port}
"""

    # Create the script file
    script_filename = "ssh_connection.sh"
    with open(script_filename, "w") as script_file:
        script_file.write(script_content)

    # Serve the file
    return send_file(script_filename, as_attachment=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=22001)
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()

    app.run(host='0.0.0.0', port=args.port, debug=args.debug)
