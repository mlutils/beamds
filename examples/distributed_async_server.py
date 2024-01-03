from src.beam.misc import BeamFakeAlg
from src.beam.distributed import AsyncServer
from src.beam.logger import beam_logger as logger


if __name__ == '__main__':
    # Create a fake algorithm
    fake_alg = BeamFakeAlg(sleep_time=10., variance=0.5, error_rate=0.1)

    def postrun(task_args=None, **kwargs):
        logger.info(f'Task has completed for {task_args} with {kwargs}')

    server = AsyncServer(fake_alg, postrun=postrun, port=36751, ws_port=36703,
                         )
    server.run()
