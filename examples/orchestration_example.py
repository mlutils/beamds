from typing import List
from src.beam import beam_logger as logger
from src.beam.orchestration import BeamK8S
from kubernetes import client, config
from src.beam.orchestration.k8s import BeamDeploy, ServiceConfig

api_url = "https://api.kh-dev.dt.local:6443"
api_token = "sha256~6fzKeFQhEO2I4skt5Bj07epEiYsZclZIvXsrD0TiwRk"
project_name = "ben-guryon"
image_name = "harbor.dt.local/public/jup:02"
#ports: list[int] = [22, 8888]
labels = {"app": "bgu"}
deployment_name = "bgu"
namespace = "ben-guryon"
replicas = 1
entrypoint_args = ["63"]  # Container arguments
entrypoint_envs = {"TEST": "test"}  # Container environment variables
#service_type = "NodePort" # Service types are: ClusterIP, NodePort, LoadBalancer, ExternalName
use_scc = True  # Pass the SCC control parameter
service_configs = [ # Service types are: ClusterIP, NodePort, LoadBalancer, ExternalName
    ServiceConfig(port=22, service_name="ssh", service_type="NodePort", port_name="ssh-port"),
    ServiceConfig(port=8888, service_name="jupyter", service_type="LoadBalancer", port_name="jupyter-port", create_route=True, route_protocol='http'),
]

print('hello world')
print("API URL:", api_url)
print("API Token:", api_token)

# the order of the VARS is important!!! (see BeamK8S class)
k8s = BeamK8S(
    api_url=api_url,
    api_token=api_token,
    project_name=project_name,
    namespace=namespace,
)

deployment = BeamDeploy(
    k8s=k8s,
    project_name=project_name,
    namespace=namespace,
    replicas=replicas,
    labels=labels,
    image_name=image_name,
    deployment_name=deployment_name,
    service_configs=service_configs,
    #ports=ports,
    use_scc=use_scc,  # Pass the SCC control parameter
    scc_name="anyuid",
    entrypoint_args=entrypoint_args,
    entrypoint_envs=entrypoint_envs,
    #service_type=service_type,
)

deployment.launch(replicas=1)

print("Fetching external endpoints...")
internal_endpoints = k8s.get_internal_endpoints_with_nodeport(namespace=namespace)
for endpoint in internal_endpoints:
    print(f"Internal Access: {endpoint['node_ip']}:{endpoint['node_port']}")




# deployment = BeamDeploy(
#     k8s=k8s,
#     labels=labels,
#     image_name=image_name,
#     deployment_name=deployment_name,
#     ports=ports,
#     entrypoint_envs=entrypoint_envs
# )
#
# deployment.launch(replicas=1)


# k8s = BeamK8S(
#     api_url=api_url,
#     api_token=api_token,
# )
#
# pod = BeamPod(
#     k8s=k8s,
#     project_name=project_name,
#     namespace=namespace,
#     replicas=1,  # Ensure this is an integer
#     labels=labels,
#     image_name=image_name,
#     deployment_name=deployment_name,
#     ports=ports,
#     env=env)



# k8s.get_pods(namespace="kleiner")
# for ns in k8s.namespaces.items:
#     logger.info(f"Namespace: {ns.metadata.name}")
#
# k8s.get_deployments(namespace="kleiner")




#k8s = BeamK8S(api_url="https://api.kh-dev.dt.local:6443", api_token="sha256~rcBAFKblEq83dkshTqhMeCa8Gl2kUSHP4QunM8ukxJI")

# k8s.get_pods(namespace="kleiner")
#
# for ns in k8s.namespaces.items:
#     logger.info(f"Namespace: {ns.metadata.name}")
#
# k8s.get_deployments(namespace="kleiner")
#print(deployment)

# api_instance = client.AppsV1Api()
#
# api_response = api_instance.create_namespaced_deployment(
#     body=deployment,
#     namespace=namespace
# )
#
# print("Deployment created. status='%s'" % str(api_response.status))


#k8s.create_deployment(namespace="kleiner", deployment_name="yoss", image="beam", replicas=1, port=80, cpu_request=0.1, cpu_limit=1)


# k8s.generate_yaml_config('/tmp/my_conf.yaml', namespace='kleiner', deployment_name='yoss', image='beam',
#                          replicas=1, port=80, cpu_request=0.1, cpu_limit=1,)
#
# k8s.generate_file_config('/tmp/my_conf.json', namespace='kleiner', deployment_name='yoss', image='beam',
#

#k8s.get_cpu_resources(namespace="kleiner", deployment_name="Yos")

