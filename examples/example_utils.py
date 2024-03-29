import os
import sys
import torch
import numpy as np
from src.beam import resource


def parallel_treading():

    from src.beam.parallel import parallel, task

    def func(i):
        print(f'func {i}\n')
        return i ** 2

    res = parallel([task(func)(i) for i in range(10)], n_workers=2, method='threading')

    print(res)


def comet_path():
    path = resource('comet:///eladsar/')
    p = path.joinpath('beam', 'yyyy')
    # p.joinpath('a.pt').write(torch.randn(10))
    # p.joinpath('a.pt').read(ext='.pt')

    # p.joinpath('a.pkl').write({'a': 1, 'b': 2})
    # print(p.joinpath('a.pkl').read())

    # p.joinpath('a.npy').write(np.random.randn(10))
    # print(p.joinpath('a.npy').read())

    p.joinpath('a.json').write({'a': 1, 'b': 2})
    print(p.joinpath('a.json').read())
    print('done!')


def add_beam_to_path():
    beam_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../src')
    sys.path.insert(0, beam_path)


add_beam_to_path()


def build_hparams():
    from src.beam.tabular import TabularHparams
    hparams = TabularHparams(path_to_data='xxxxx')

    print(hparams.__dict__)
    print(hparams)


def text_beam_data():

    from src.beam import BeamData

    bd = BeamData.from_path('/tmp/example')

    # print(bd.values)
    # print(bd.stacked_values)

    print(bd)

    bd['c'] = np.random.randn(100)
    bd['d'] = np.random.randn(100)

    bd.cache()

    print(bd)

    bd.store()
    print(bd)


def write_bundle_tabular(path):

    from src.beam import beam_logger as logger

    from src.beam.tabular import TabularTransformer, TabularHparams, DeepTabularAlg

    kwargs_base = dict(algorithm='debug_reporter',
                       # path_to_data='/dsi/shared/elads/elads/data/tabular/dataset/data/',
                       path_to_data='/home/dsi/elads/data/tabular/data/',
                       path_to_results='/dsi/shared/elads/elads/data/tabular/results/',
                       copy_code=False, dynamic_masking=False, comet=False, tensorboard=True, n_epochs=2,
                       stop_at=0.98, parallel=1, device=1, n_quantiles=6, label_smoothing=.2)

    kwargs_all = {}

    k = 'california_housing'
    kwargs_all[k] = dict(batch_size=128)

    logger.info(f"Starting a new experiment with dataset: {k}")
    hparams = {**kwargs_base}
    hparams.update(kwargs_all[k])
    hparams['dataset_name'] = k
    hparams['identifier'] = k
    hparams = TabularHparams(hparams)

    # exp = Experiment(hparams)
    # dataset = TabularDataset(hparams)
    net = TabularTransformer(hparams, 10, [4, 4, 4], [0, 0, 1])
    alg = DeepTabularAlg(hparams, networks=net)

    from src.beam.auto import AutoBeam

    ab = AutoBeam(alg)

    # print(ab.module_spec)
    # print(ab.module_walk)
    # print(ab.module_dependencies)
    # print(ab.requirements)
    # print(ab.top_levels)
    # print(ab.module_to_tar(path))
    print(ab.private_modules)

    # autobeam_path = '/tmp/autobeam'
    # beam_path(autobeam_path).rmtree()

    # ab.module_to_tar(tar_path)
    # AutoBeam.to_bundle(alg, autobeam_path)


def write_bundle_cifar(path):

    from src.beam import beam_arguments
    from cifar10_example import CIFAR10Algorithm
    from src.beam.auto import AutoBeam

    path.rmtree()

    args = beam_arguments(
        f"--project-name=cifar10 --algorithm=CIFAR10Algorithm --device=1 --half --lr-d=1e-4 --batch-size=512",
        "--n-epochs=50 --epoch-length-train=50000 --epoch-length-eval=10000 --clip=0 --parallel=1 --accumulate=1 --no-deterministic",
        "--weight-decay=.00256 --momentum=0.9 --beta2=0.999 --temperature=1 --objective=acc --scheduler=one_cycle",
        dropout=.0, activation='gelu', channels=512, label_smoothing=.2, padding=4, scale_down=.7,
        scale_up=1.4, ratio_down=.7, ratio_up=1.4)

    alg = CIFAR10Algorithm(args)
    # ab = AutoBeam(alg)

    # print(ab.requirements)
    # print(ab.private_modules)
    # print(ab.import_statement)
    #
    # ab.modules_to_tar(path.joinpath('modules.tar.gz'))
    AutoBeam.to_bundle(alg, path)
    print('done writing bundle')


def load_bundle(path):
    from src.beam.auto import AutoBeam
    alg = AutoBeam.from_path(path)
    print(alg)
    print(alg.hparams)
    print('done loading bundle')
    return alg


def test_data_apply():

    M = 40000

    nel = 100
    k1 = 20000
    k2 = 20

    def gen_coo_vectors(k):
        r = []
        c = []
        v = []

        for i in range(k):
            r.append(i * torch.ones(nel, dtype=torch.int64))
            c.append(torch.randint(M, size=(nel,)))
            v.append(torch.randn(nel))

        return torch.sparse_coo_tensor(torch.stack([torch.cat(r), torch.cat(c)]), torch.cat(v), size=(k, M))

    s1 = gen_coo_vectors(k1)
    s2 = gen_coo_vectors(k2)
    s3 = gen_coo_vectors(k2)

    from src.beam import BeamData
    from uuid import uuid4 as uuid
    from beam.server.beam_client import BeamClient

    # sparse_sim = SparseSimilarity(metric='cosine', format='coo', vec_size=10000, device='cuda', k=10)
    sparse_sim = BeamClient('localhost:27451')

    sparse_sim.add(s1)
    print(sparse_sim.k)

    tasks = [{'req_id': str(uuid()), 'arg': s2}, {'req_id': str(uuid()), 'arg': s3}]
    bd = BeamData.simple({t['req_id']: t['arg'] for t in tasks})
    bd2 = bd.apply(sparse_sim.search)
    print(bd)
    print(bd.values)
    print(tasks)

    results = {task['req_id']: bd2[task['req_id']] for task in tasks}
    print(results)


if __name__ == '__main__':

    # path = '/home/dsi/elads/sandbox/cifar10_bundle'
    # path = beam_path(path)
    #
    # # write_bundle_cifar(path)
    #
    # alg = load_bundle(path)


    # from src.beam.config import BeamHparams
    # from src.beam.tabular import TabularHparams
    #
    # hparams = BeamHparams(identifier='test', project_name='test', algorithm='test', device=1)
    #
    # # hparams = TabularHparams(identifier='test', project_name='test', algorithm='test', device=1)
    # hparams = TabularHparams(hparams)
    # print(hparams)

    # test_data_apply()

    # text_beam_data()

    # build_hparams()

    # comet_path()

    parallel_treading()

    print('done')