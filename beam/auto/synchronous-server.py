
import beam
from beam.serve import beam_server, BeamServeConfig
from beam import resource
from beam.auto import AutoBeam
from beam import beam_logger as logger


if __name__ == '__main__':

    logger.info(f"Starting Beam Serve: beam version: {beam.__version__}")
    config = BeamServeConfig()
    path_to_bundle = resource(config.path_to_bundle)
    logger.info(f"Loading bundle from: {path_to_bundle}")

    obj = AutoBeam.from_bundle(path_to_bundle)

    logger.info(f"Starting Beam with parameters: {config}")
    beam_server(obj, protocol=config.protocol, port=config.serve_port, n_threads=config.n_threads,
                use_torch=config.use_torch, batch=config.batch, tls=config.tls,
                max_batch_size=config.max_batch_size, max_wait_time=config.max_wait_time,
                backend=config.http_backend)
