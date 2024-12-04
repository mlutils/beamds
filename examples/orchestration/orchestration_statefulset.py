from beam.orchestration import (BeamK8S, BeamStatefulSet)
from beam.logging import beam_logger as logger
import os
import time

from beam.resources import resource, this_dir
from beam.orchestration.config import K8SConfig

# Initialize Beam configuration
script_dir = os.path.dirname(os.path.realpath(__file__))
conf_path = resource(os.path.join(script_dir, 'orchestration_beamstatefulset.yaml')).str
config = K8SConfig(conf_path)

# Print API details
print("API URL:", config['api_url'])
print("API Token:", config['api_token'])

# Initialize Beam Kubernetes instance
k8s = BeamK8S(
    api_url=config['api_url'],
    api_token=config['api_token'],
    project_name=config['project_name'],
    namespace=config['project_name'],
)

# Initialize and launch the StatefulSet
st = BeamStatefulSet(config, k8s)
st.launch()
print("StatefulSet launched, waiting for 15 seconds to get logs...")
time.sleep(15)
logs = st.get_statefulset_logs()
print(f"Logs: {logs}")

print("\n Try cleanup")
k8s.cleanup_statefulsets(
    namespace=config['project_name'],
    app_name=config['statefulset_name'],
)