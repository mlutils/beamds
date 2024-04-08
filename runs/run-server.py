import beam_source

import beam
from beam.serve import beam_server
from beam.config import BeamHparams, BeamParam
from beam import resource
from beam.auto import AutoBeam
from beam import beam_logger as logger


class BeamServeConfig(BeamHparams):

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


if __name__ == '__main__':

    logger.info(f"Starting Beam Serve: beam version: {beam.__version__}")
    config = BeamServeConfig()
    path_to_bundle = resource(config.path_to_bundle)
    logger.info(f"Loading bundle from: {path_to_bundle}")

    obj = AutoBeam.from_bundle(path_to_bundle)

    logger.info(f"Starting Beam with parameters: {config}")
    beam_server(obj, protocol=config.protocol, port=config.port, n_threads=config.n_threads,
                use_torch=config.use_torch, batch=config.batch, tls=config.tls,
                max_batch_size=config.max_batch_size, max_wait_time=config.max_wait_time,
                backend=config.http_backend)
