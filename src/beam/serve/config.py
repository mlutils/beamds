
from ..config import BeamConfig, BeamParam


class BeamServeConfig(BeamConfig):

    defaults = {}
    parameters = [
        BeamParam('protocol', str, 'http', 'The serving protocol [http|grpc]', model=False),
        BeamParam('http-backend', str, 'waitress', 'The HTTP server backend', model=False),
        BeamParam('path-to-bundle', str, '/workspace/serve/bundle', 'Where the algorithm bundle is stored', model=False),
        BeamParam('port', int, None, 'Default port number (set None to choose automatically)', model=False),
        BeamParam('n-threads', int, 4, 'parallel threads', model=False),
        BeamParam('use-torch', bool, False, 'Whether to use torch for pickling/unpickling', model=False),
        BeamParam('batch', str, None, 'A function to parallelize with batching', model=False),
        BeamParam('tls', bool, True, 'Whether to use tls encryption', model=False),
        BeamParam('max-batch-size', int, 10, 'Maximal batch size (execute function when reaching this number)', model=False),
        BeamParam('max-wait-time', float, 1., 'execute function if reaching this timeout', model=False),
    ]