from beam.misc import BeamFakeAlg
from beam.serve import BeamServeConfig
from beam.serve import beam_server


def main():
    fake_alg = BeamFakeAlg(sleep_time=1., variance=0.5, error_rate=0.1)
    config = BeamServeConfig()
    beam_server(fake_alg, protocol=config.protocol, port=config.port, n_threads=config.n_threads,
                use_torch=config.use_torch, batch=config.batch, tls=config.tls,
                max_batch_size=config.max_batch_size, max_wait_time=config.max_wait_time,
                backend=config.http_backend)


if __name__ == '__main__':
    main()
