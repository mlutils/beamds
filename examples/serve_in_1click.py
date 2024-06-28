from beam.misc.fake import BeamFakeAlg
from beam.serve import BeamServeConfig
from beam.orchestration import HTTPServeCluster
from beam.resources import resource
from beam.auto import AutoBeam
import os

# https://github.com/mlutils/beamds/blob/dev/notebooks/to-docker-example.ipynb


fake_alg = BeamFakeAlg(sleep_time=1)

print(fake_alg.run(123))

# from src.beam.serve import beam_server
# beam_server(fake_alg)

config = BeamServeConfig(port=44044, **{'path-to-bundle': '/app/algorithm'})


AutoBeam.to_docker(obj=fake_alg, base_image='eladsar/beam:20240605', image_name='fake-alg-http-server',
                   beam_version='2.5.11', config=config, push_image=True,
                   registry_url='harbor.dt.local', username='admin', password='Har@123')


# script_dir = os.path.dirname(os.path.realpath(__file__))
# conf_path = resource(os.path.join(script_dir, 'orchestration_configuration.json')).str
# config = HTTPServeCluster(conf_path)

# from beam.orchestration.cluster import HTTPServeCluster
#
# HTTPServeCluster.from_algorithm(fake_alg)
# Usage example
# image_manager = ImageManager()
# image_manager._push_image('my-image:latest', 'http://myregistry.example.com:5000', username='myuser', password='mypass',
# #                           insecure_registry=True)
# # Usage example
#     _push_image('my-image:latest', 'http://myregistry.example.com:5000', username='myuser', password='mypass',
#                 insecure_registry=True)
