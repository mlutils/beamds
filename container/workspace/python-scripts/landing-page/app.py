from flask import Flask, render_template, url_for, send_file
import argparse

app = Flask(__name__)

@app.route('/')
def home():
    services = [
        {"name": "MLflow", "icon": url_for('static', filename='mlflow_icon.png'), "url": "http://localhost:your_mlflow_port", "description": "MLflow is an open-source platform for managing the end-to-end machine learning lifecycle."},
        {"name": "Jupyter", "icon": url_for('static', filename='jupyter_icon.png'), "url": "http://localhost:your_jupyter_port", "description": "Jupyter Notebook is an open-source web application that allows you to create and share documents that contain live code, equations, visualizations, and narrative text."},
        {"name": "SSH", "icon": url_for('static', filename='ssh_icon.png'), "url": url_for('generate_ssh_script'), "description": "SSH (Secure Shell) is a protocol for securely accessing network services over an unsecured network."},
        {"name": "Ray", "icon": url_for('static', filename='ray_icon.png'), "url": "http://localhost:your_ray_dashboard_port", "description": "Ray is a distributed computing framework for parallel and distributed Python."},
        {"name": "Chroma", "icon": url_for('static', filename='chroma_icon.png'), "url": "http://localhost:your_chroma_port", "description": "Chroma is a vector database for building high-performance machine learning applications."},
        {"name": "Redis", "icon": url_for('static', filename='redis_icon.png'), "url": "http://localhost:your_redis_port", "description": "Redis is an in-memory data structure store used as a database, cache, and message broker."},
        {"name": "RabbitMQ", "icon": url_for('static', filename='rabbitmq_icon.png'), "url": "http://localhost:your_rabbitmq_port", "description": "RabbitMQ is an open-source message broker software that implements the Advanced Message Queuing Protocol (AMQP)."},
        {"name": "Prefect", "icon": url_for('static', filename='prefect_icon.png'), "url": "http://localhost:your_prefect_port", "description": "Prefect is a workflow orchestration tool that helps you to coordinate and monitor the execution of data workflows."},
        {"name": "MongoDB", "icon": url_for('static', filename='mongo_icon.png'), "url": "http://localhost:your_mongo_port", "description": "MongoDB is a source-available cross-platform document-oriented database program."},
        {"name": "MinIO", "icon": url_for('static', filename='minio_icon.png'), "url": "http://localhost:your_minio_port", "description": "MinIO is a high-performance, S3 compatible object storage system."},
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
