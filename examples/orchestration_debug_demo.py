from beam import resource
from beam.serve import beam_server
from beam import deploy_server
from beam import HTTPServeClusterConfig


alg = resource('/tmp/yolo-bundle/').read(ext='.abm')
beam_server(alg, non_blocking=True)
conf = HTTPServeClusterConfig('/home/beamds/examples/orchestration_beamdemo.yaml')
deploy_server('/tmp/yolo-bundle/', conf)
