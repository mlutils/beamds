from beam.server.beam_server import BeamServer

try:
    import torch

    has_torch = True
except ImportError:
    has_torch = False


def beam_remote(obj, host=None, port=None, debug=False):
    server = BeamServer(obj)
    server.run(host=host, port=port, debug=debug)
