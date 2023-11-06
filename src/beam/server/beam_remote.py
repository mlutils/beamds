from .beam_server import BeamServer


def beam_remote(obj, host=None, port=None, debug=False):
    server = BeamServer(obj)
    server.run(host=host, port=port, debug=debug)
