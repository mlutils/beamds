from src.beam import beam_logger as logger
from src.beam.orchestration import BeamK8S
print('hello world')
k8s = BeamK8S(api_url="https://api.kh-dev.dt.local:6443", api_token="sha256~5BjsBmhuK-bFa5shjVAqkRqLAIpUTxHXbIqE756L9NY")
print("API URL:", k8s.api_url)
print("API Token:", k8s.api_token)

#k8s.client()
k8s.get_pods(namespace="kleiner")

for ns in k8s.namespaces.items:
    logger.info(f"Namespace: {ns.metadata.name}")

k8s.get_deployments(namespace="kleiner")

k8s.generate_yaml_config('/tmp/my_conf.yaml', namespace='kleiner', deployment_name='yoss', image='beam',
                         replicas=1, port=80, cpu_request=0.1, cpu_limit=1,)

k8s.generate_file_config('/tmp/my_conf.json', namespace='kleiner', deployment_name='yoss', image='beam',
                         replicas=1, port=80, cpu_request=0.1, cpu_limit=1,)


#k8s.get_cpu_resources(namespace="kleiner", deployment_name="Yos")

