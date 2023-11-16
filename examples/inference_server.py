from examples.example_utils import add_beam_to_path
add_beam_to_path()

from src.beam.serve import BeamServer
# from examples.ssl_with_beam import my_ssl_algorithm
from src.beam.sparse import SparseSimilarity


if __name__ == '__main__':

    # Alg = my_ssl_algorithm('BeamVICReg')
    # serve = BeamServer.build_algorithm_from_path("/home/shared/data/results/beam_ssl/BeamVICReg/resnet_parallel/0000_20220823_132503", Alg,
    #                                               override_hparams={'device': 2, 'lgb_device': None,
    #                                                                 'lgb_rounds': 40, 'lgb_num_leaves': 31,
    #                                                                 'lgb_max_depth': 4,
    #                                                                 'verbose_lgb': False})

    M = 40000
    sparse_sim = SparseSimilarity(metric='cosine', format='coo', vec_size=M, device='cuda', k=10)
    # server = BeamServer(sparse_sim, max_batch_size=2, max_wait_time=10, batch='search')
    server = BeamServer(sparse_sim)
    server.run(server='waitress')