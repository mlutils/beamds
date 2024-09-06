from beam.orchestration import BeamManager
from beam.serve.http_server import HTTPServer
import openshift
import docker
import waitress
import namegenerator


class BeamManagerWrapper(BeamManager):
    pass
