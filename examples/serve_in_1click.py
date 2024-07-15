from beam.orchestration import (HTTPServeCluster, HTTPServeClusterConfig)
from beam.resources import resource
from beam.misc.fake import BeamFakeAlg
import os

script_dir = os.path.dirname(os.path.realpath(__file__))

conf_path = resource(os.path.join(script_dir, 'Server_in_1click_config.yaml')).str

config = HTTPServeClusterConfig(conf_path, conf_bundle=True, port=44044, **{'path-to-bundle': '/app/algorithm'})

print('hello world')
print("API URL:", config.api_url)
print("API Token:", config.api_token)


alg = BeamFakeAlg(sleep_time=1)
# image_name = "harbor.dt.local/public/fake-alg-http-server:latest"
image_name = None
# to_email = input("Enter the email address to receive the cluster info: ")
to_email = 'yossi@dayo-tech.com'


# serve_cluster = HTTPServeCluster(deployment=None, alg=alg, config=config, pods=config.pods,
#                                  base_url=config.base_url, to_email=config.to_email, send_mail=config.send_email,
#                                  registry_project_name=config.registry_project_name,)

# HTTPServeCluster.deploy_from_bundle('/app/algorithm', config)
# serve_cluster.deploy_from_algorithm(alg, config)
HTTPServeCluster.deploy_from_algorithm(alg, config)
# HTTPServeCluster.deploy_from_image(image_name, config)









# if serve_cluster:
#     # print(serve_cluster.deployment.cluster_info())
# else:
#     print("Deployment failed.")
# print(serve_cluster.deployment.cluster_info)

# https://github.com/mlutils/beamds/blob/dev/notebooks/to-docker-example.ipynb


# fake_alg = BeamFakeAlg(sleep_time=1)
#
# print(fake_alg.run(123))
#
# # from src.beam.serve import beam_server
# # beam_server(fake_alg)
#
# config = BeamServeConfig(port=44044, **{'path-to-bundle': '/app/algorithm'})


# AutoBeam.to_docker(obj=fake_alg, base_image='eladsar/beam:20240605', image_name='fake-alg-http-server',
#                    beam_version='2.5.11', config=config, push_image=True,
#                    registry_url='harbor.dt.local', username='admin', password='Har@123')





#
# HTTPServeCluster.from_algorithm(fake_alg)
# Usage example
# image_manager = ImageManager()
# image_manager._push_image('my-image:latest', 'http://myregistry.example.com:5000', username='myuser', password='mypass',
# #                           insecure_registry=True)
# # Usage example
#     _push_image('my-image:latest', 'http://myregistry.example.com:5000', username='myuser', password='mypass',
#                 insecure_registry=True)
