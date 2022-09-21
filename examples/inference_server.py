from examples.example_utils import add_beam_to_path
add_beam_to_path()

from src.beam.server import BeamServer
from examples.ssl_with_beam import my_ssl_algorithm


if __name__ == '__main__':

    Alg = my_ssl_algorithm('BeamVICReg')
    server = BeamServer.build_algorithm_from_path("/home/shared/data/results/beam_ssl/BeamVICReg/resnet_parallel/0000_20220823_132503", Alg,
                                                  override_hparams={'device': 2, 'lgb_device': None,
                                                                    'lgb_rounds': 40, 'lgb_num_leaves': 31,
                                                                    'lgb_max_depth': 4,
                                                                    'verbose_lgb': False})
    server.run(debug=False)
