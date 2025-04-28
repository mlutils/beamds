
from ..config import BeamConfig, BeamParam


class BeamServeConfig(BeamConfig):

    defaults = {}
    parameters = [
        BeamParam('protocol', str, 'http', 'The serving protocol [http|grpc]'),
        BeamParam('http-backend', str, 'waitress', 'The HTTP server backend'),
        BeamParam('path-to-bundle', str, '/app/algorithm', 'Where the algorithm bundle is stored'),
        BeamParam('path-to-state', str, None, 'Where the state is stored (has precedence over path-to-bundle)'),
        BeamParam('serve_port', int, 35000, 'Default port number (set None to choose automatically)'),
        BeamParam('n-threads', int, 4, 'parallel threads'),
        BeamParam('use-torch', bool, False, 'Whether to use torch for pickling/unpickling'),
        BeamParam('batch', str, None, 'A function to parallelize with batching'),
        BeamParam('tls', bool, False, 'Whether to use tls encryption'),
        BeamParam('max-batch-size', int, 10, 'Maximal batch size (execute function when reaching this number)'),
        BeamParam('max-wait-time', float, 1., 'execute function if reaching this timeout'),
        BeamParam('non-blocking', bool, False, 'Run the server in a non-blocking mode'),
        BeamParam('load-kwargs', dict, {}, 'Additional kwargs to pass to the load function'),
    ]