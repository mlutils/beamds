from beam.orchestration import (BeamK8S, BeamStatefulSet)
from beam.logging import beam_logger as logger
import time
from beam.resources import resource, this_dir
from beam.orchestration.config import K8SConfig
import os
import sys


script_dir = os.path.dirname(os.path.realpath(__file__))
conf_path = resource(os.path.join(script_dir, 'orchestration_beamdeploy.yaml')).str
config = K8SConfig(conf_path)


print('hello world')
print("API URL:", config['api_url'])
print("API Token:", config['api_token'])


# the order of the VARS is important!! (see BeamK8S class)
k8s = BeamK8S(
    api_url=config['api_url'],
    api_token=config['api_token'],
    project_name=config['project_name'],
    namespace=config['project_name'],
)

print(f"k8s type: {type(k8s)}")
print(dir(k8s))
print([attr for attr in dir(k8s) if callable(getattr(k8s, attr))])




st = BeamStatefulSet(config, k8s)

# Launch deployment and obtain pod instances
st.launch(replicas=1)
logger.debug(f"Home-Page: {st.k8s.get_homepage_route_url(namespace=config['project_name'])}")