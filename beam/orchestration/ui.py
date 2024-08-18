from flask import Flask, request, Response, jsonify
import yaml
import os  # To use environment variables
from kubernetes import client
from openshift.dynamic import DynamicClient
from beam.orchestration import BeamK8S, BeamDeploy

app = Flask(__name__)


def get_config():
    # Get config from environment variables or set defaults
    api_url = os.getenv('API_URL', 'https://api.kh-dev.dt.local:6443')
    api_token = os.getenv('API_TOKEN', 'eyJhbGciOiJSUzI1NiIsImtpZCI6Imhtdk5nbTRoenVRenhkd0lWdnBWMUI0MmV2ZGpxMk8wQ0NaMlhmejZBc1UifQ.eyJpc3MiOiJrdWJlcm5ldGVzL3NlcnZpY2VhY2NvdW50Iiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9uYW1lc3BhY2UiOiJzdGFyZmllbGQiLCJrdWJlcm5ldGVzLmlvL3NlcnZpY2VhY2NvdW50L3NlY3JldC5uYW1lIjoic3RhcmZpZWxkc3ZjLXRva2VuLWZid2hzIiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9zZXJ2aWNlLWFjY291bnQubmFtZSI6InN0YXJmaWVsZHN2YyIsImt1YmVybmV0ZXMuaW8vc2VydmljZWFjY291bnQvc2VydmljZS1hY2NvdW50LnVpZCI6ImMwMWFiN2E2LTMxNDEtNDFjNi04ZTgyLWIzYjY5MjYwNTY3YyIsInN1YiI6InN5c3RlbTpzZXJ2aWNlYWNjb3VudDpzdGFyZmllbGQ6c3RhcmZpZWxkc3ZjIn0.kWOkxF7vD9g5PDzv7nlNIVYm1dYDAXve1MH-HzfWXBrrcUZCSI3Wu_IH_PvBhshphvbPBFJrCnLciFMC0Mt0C54vE1T6G6_cpqSThvPTmpyh0fORLhxQ98jIXfY40yuTDYyOZgefHsZJ2aGUSDPQlzSsT8PLBfbaVw9Y7aqimc70EUKEREmUY8NHhhttw0pNUCfB0WkZpSRUW9gLZjdZgwOyKLnhsIoErS5I9sIc7aILyjvOVpewZKI0KBU5aUi9jZSzbI5aXycx3P2YBp4KCOWqdL8AYVlMDcfSFX1c5yRa8LhLciFGzop9lT6HiDHWGzkiWhcYubGkxE3pLOfXMI9WoW5FWzTaXMeL6qNxenvAU9rbUkjbtxTsQ_sursEMCgcWgW8Tm5jbYrZDYVeH71l4ph7X-CtPZ7zJ7dIwxlSJ_seJ7Ardkk6LgB24Po20raizX063PAd5ivqYCLOL3jfN0wFReRX_TZMID47gNmyyG-_p6vuKublKNcjqFuoKY7Cw30jQp6bxSzDmYecwT7wFOn3gyOyUUbKj7auycIZxPJSjsR-TdLl9TyrSzSSsQV-Sg2JqPA15U9E2-T48m1RLz2ckyOR5LEsrhx9ESJTc1qcry6Ho4x1cG9nMVYfg4uZey_6CXH0giUtM3-5pt5vUKb746GbUIoend7yMCmI (∫orchestration_tokens.py:<module>-#43)')
    namespace = os.getenv('NAMESPACE', 'starifeld')
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
    app.run(host='0.0.0.0', port=64098)
