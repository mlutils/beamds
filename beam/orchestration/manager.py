from flask import Flask, request, Response, jsonify
import yaml
import os  # To use environment variables
from kubernetes import client
from openshift.dynamic import DynamicClient
from beam.orchestration import BeamK8S, BeamDeploy

app = Flask(__name__)


def get_config():
    # Get config from environment variables or set defaults
    api_url = os.getenv('API_URL', 'http://default-url')
    api_token = os.getenv('API_TOKEN', 'default-token')
    namespace = os.getenv('NAMESPACE', 'default-namespace')
    return api_url, api_token, namespace


def init_clients(api_url, api_token, namespace):
    configuration = client.Configuration()
    configuration.host = api_url
    configuration.verify_ssl = False
    configuration.debug = False
    configuration.api_key = {'authorization': f"Bearer {api_token}"}

    k8s_client_instance = client.ApiClient(configuration)
    k8s_client = BeamK8S(api_client=k8s_client_instance, namespace=namespace)

    ocp_client = DynamicClient(k8s_client_instance)

    return k8s_client, ocp_client


@app.route('/')
def home():
    endpoints = {rule.endpoint: {
        "url": rule.rule,
        "methods": list(rule.methods - {'HEAD', 'OPTIONS'}),
        "description": app.view_functions[rule.endpoint].__doc__
    } for rule in app.url_map.iter_rules() if rule.endpoint != 'static'}

    return jsonify({
        'message': 'Welcome to the BeamK8S Manager API',
        'endpoints': endpoints,
        'instructions': 'Set API_URL, API_TOKEN, and NAMESPACE as environment variables or pass them in requests.'
    })


@app.route('/create_deployment', methods=['POST'])
def create_deployment():
    """Create a Kubernetes deployment based on provided YAML configuration."""
    data = yaml.safe_load(request.data) if request.data else {}
    api_url, api_token, namespace = (
        data.get('api_url') or request.args.get('api_url'),
        data.get('api_token') or request.args.get('api_token'),
        data.get('namespace') or request.args.get('namespace')
    )

    if not all([api_url, api_token, namespace]):
        api_url, api_token, namespace = get_config()

    k8s_client, ocp_client = init_clients(api_url, api_token, namespace)
    beam_deploy = BeamDeploy(data, k8s_client)
    result = beam_deploy.launch(replicas=data.get('replicas'))
    return Response(yaml.dump({"status": "success", "result": str(result)}), mimetype='application/yaml')


@app.route('/get_pod_info', methods=['GET'])
def get_pod_info():
    """Retrieve information about a specific pod using its name and namespace."""
    api_url, api_token, namespace = (
        request.args.get('api_url'),
        request.args.get('api_token'),
        request.args.get('namespace')
    )

    if not all([api_url, api_token, namespace]):
        api_url, api_token, namespace = get_config()

    k8s_client, ocp_client = init_clients(api_url, api_token, namespace)
    pod_name = request.args.get('pod_name')
    pod_info = k8s_client.get_pod_info(pod_name, namespace)
    return Response(yaml.dump({"status": "success", "pod_info": pod_info}), mimetype='application/yaml')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
