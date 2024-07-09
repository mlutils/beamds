from beam import resource
import beam
from beam.serve import beam_server, BeamServeConfig
from beam import logger


if __name__ == '__main__':
    logger.info(f"Starting Beam Server: beam version: {beam.__version__}")
    config = BeamServeConfig()

    if config.path_to_state is not None:
        logger.info(f"Loading state from {config.path_to_state} (assuming code+requirements are already loaded)")
        obj = resource(config.path_to_state).read(ext='.bmpr', **config.load_kwargs)
    else:
        logger.info(f"Loading algorithm bundle (code+state+requirements) from {config.path_to_bundle}")
        obj = resource(config.path_to_bundle).read(ext='.abm', **config.load_kwargs)

    logger.info(f"Starting Beam Server with parameters: {config}")
    beam_server(obj, **config)
