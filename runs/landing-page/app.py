from flask import Flask, render_template
import os

app = Flask(__name__)

# Define the services and their icons
services = [
    {"name": "MLflow", "icon": "path/to/mlflow_icon.png", "url": "http://localhost:your_mlflow_port"},
    {"name": "Jupyter", "icon": "path/to/jupyter_icon.png", "url": "http://localhost:your_jupyter_port"},
    {"name": "SSH", "icon": "path/to/ssh_icon.png", "url": "http://localhost:your_ssh_port"},
    {"name": "Redis", "icon": "path/to/redis_icon.png", "url": "http://localhost:your_redis_port"},
    {"name": "RabbitMQ", "icon": "path/to/rabbitmq_icon.png", "url": "http://localhost:your_rabbitmq_port"},
    {"name": "Prefect", "icon": "path/to/prefect_icon.png", "url": "http://localhost:your_prefect_port"},
    {"name": "Ray", "icon": "path/to/ray_icon.png", "url": "http://localhost:your_ray_dashboard_port"},
    {"name": "Chroma", "icon": "path/to/chroma_icon.png", "url": "http://localhost:your_chroma_port"},
    {"name": "MongoDB", "icon": "path/to/mongo_icon.png", "url": "http://localhost:your_mongo_port"},
    {"name": "MinIO", "icon": "path/to/minio_icon.png", "url": "http://localhost:your_minio_port"},
]

# Root icon for the entire image
root_icon = "path/to/root_icon.png"

@app.route('/')
def home():
    return render_template('index.html', services=services, root_icon=root_icon)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)