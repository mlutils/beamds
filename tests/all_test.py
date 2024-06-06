import os
import sys
import time
from collections import Counter

import torch
import numpy as np
from src.beam import resource, BeamData
from src.beam import beam_logger as logger
import pandas as pd


def test_beam_default_configuration():
    import sys
    # generate some random arguments
    sys.argv = ['python', '--a', '1', '--b', '2', '--c', '3', '--d', '4', '--e', '5', '--f', '6', '--g', '7', '--h', '8']
    from src.beam import Transformer
    p = Transformer()
    print(p.hparams)


def test_ray_actor():

    from src.beam.distributed.ray_dispatcher import RayDispatcher, RayClient
    class A:
        def __init__(self, a):
            self.a = a

        def f(self, x):
            return x * self.a

    RayClient(ray_kwargs={'runtime_env': {"working_dir": "../src"}})

    Ar = RayDispatcher(A)

    ar = Ar(4)
    res = ar.f(5)

    print(res.value)


def test_special_attributes():
    from examples.enron_similarity import EnronTicketSimilarity, TicketSimilarityConfig
    from src.beam import Timer
    hparams = TicketSimilarityConfig(model_state_path='/home/shared/data/results/enron/models/model_state_e5_base',
                                     )
    alg = EnronTicketSimilarity(hparams)
    print(alg.root_path)
    with Timer():
        alg.load_state(hparams.get('model-state-path'), hparams=False, skeleton=False)

    with Timer(name='evaluate end-to-end classifier'):
        results = alg.evaluate(36, k_dense=10, k_sparse=10, threshold=0.8)

    print(results)


def test_collate_transformer_chunks():

    from src.beam import Transformer
    def func(x):
        return x + 1

    df = pd.DataFrame(data=np.random.rand(16, 4), columns=['a', 'b', 'c', 'd'])
    my_beautiful_transformer = Transformer(n_workers=1, chunksize=2, mp_method='joblib', func=func, use_dill=True)
    res = my_beautiful_transformer(df, transform_kwargs=dict(store_path='/tmp/xx'))

    add_token_transformer = Transformer(n_workers=1, chunksize=2, mp_method='joblib',
                                        func=lambda x: [xi + ' bye' for xi in x], use_dill=True)

    res = add_token_transformer(['hi how are you?', 'we are here', 'lets dance', 'it is fine'],
                                transform_kwargs=dict(store_path='/tmp/yy'))

    print(res)

    bd = BeamData.from_path('/tmp/xx')
    print(bd.stacked_values)
    print(bd)


def test_catboost():
    from sklearn.datasets import load_wine
    data = load_wine()

    x = data['data']
    y = data['target']

    from src.beam.algorithm import CBAlgorithm
    # from src.beam.config import CatboostConfig

    cb = CBAlgorithm()

    cb.fit(x, y)

def test_slice_to_index():
    from src.beam.utils import slice_to_index
    n = np.arange(10)

    print(slice_to_index(2, len(n), sliced=n))
    print(slice_to_index(slice(1), sliced=n))
    print(slice_to_index(slice(1, 3),  sliced=n))


def test_recursive_len():
    from src.beam.utils import recursive_len
    # c = Counter({'a': 1, 'b': 2})
    c = {'a': 1, 'b': 2}
    # c = {'a': 1, 'b': 2, 'c': np.random.randn(10)}
    print(recursive_len([c]))


def test_config():
    from src.beam.config import TransformerConfig

    hparams = TransformerConfig(chunksize=33333)
    hparams2 = TransformerConfig(hparams, chunksize=44444)

    print(hparams2)


def test_transformer():

    from src.beam.transformer import Transformer

    t = Transformer()
    print(t)


def test_mlflow_path():
    path = resource('mlflow:///new-exp/new-run')
    path.mkdir()
    path.joinpath('aaa.pkl').write({'a': 1, 'b': 2})
    print(path.joinpath('aaa.pkl').read())

    print(list(path))


def test_beam_data_keys():
    from src.beam import BeamData
    bd = BeamData({'a': [1, 2, 3], 'b': {'x': 1, 'y': 2}})
    print(list(bd.keys()))
    print(list(bd.keys(level=2)))
    print(list(bd.keys(level=-1)))


def simple_server():

    from src.beam.serve import beam_server
    def func(x):
        return sorted(x)

    # def func():
    #     return 'hello!'

    # def func(x):
    #     print(x)

    beam_server(func)


def grpc_server():
    from src.beam.serve.grpc_server import GRPCServer
    from src.beam.misc import BeamFakeAlg

    fake_alg = BeamFakeAlg(sleep_time=1., variance=0.5, error_rate=0.1)

    server = GRPCServer(fake_alg)
    server.run(port=28851)

    print('done!')


def distributed_client():

    alg = resource('async-http://localhost:28850')

    res = alg.run(1)
    time.sleep(3)

    print(alg.poll(res))


def distributed_server():
    from src.beam.misc import BeamFakeAlg
    from src.beam.distributed import AsyncRayServer, AsyncCeleryServer

    fake_alg = BeamFakeAlg(sleep_time=1., variance=0.5, error_rate=0.1)

    def postrun(**kwargs):
        logger.info(f'Server side callback: Task has completed for {kwargs}')

    server = AsyncRayServer(fake_alg, postrun=postrun, port=28850, ws_port=28802, asynchronous=False)
    # server = AsyncCeleryServer(fake_alg, postrun=postrun, port=28850, ws_port=28802,)

    # server.run_non_blocking()
    server.run()
    print('done!')


def sftp_example():
    import pysftp

    cnopts = pysftp.CnOpts()
    cnopts.hostkeys = None  # This disables host key checking
    path = resource('sftp://root:12345678@localhost:28822/home/elad')
    print(list(path))


def load_index():
    from src.beam.similarity import TextSimilarity
    from sklearn.datasets import fetch_20newsgroups

    logger.info(f"Loaded dataset: newsgroups_train")
    newsgroups_train = fetch_20newsgroups(subset='train')

    path = resource('/tmp/news-sim-cpu')
    text_sim = TextSimilarity.from_path(path)

    res = text_sim.nlp("Find the two closest sentences to the sentence \"It is best to estimate your loss and use it to "
                       "set your price structure\"", llm='openai:///gpt-4')

    print(res)
    # print(newsgroups_train.data[res.index[0]])
    # print(newsgroups_train.data[res.index[1]])


def save_index():
    from src.beam.similarity import TFIDF, SparnnSimilarity, DenseSimilarity, TextSimilarity
    from sklearn.datasets import fetch_20newsgroups

    logger.info(f"Loaded dataset: newsgroups_train")
    newsgroups_train = fetch_20newsgroups(subset='train')
    # newsgroups_test = fetch_20newsgroups(subset='test')

    x = newsgroups_train.data

    sim = TextSimilarity(expected_population=len(x), device=1,
                         metric='cosine', training_device='cpu', inference_device='cpu',
                         ram_footprint=2 ** 8 * int(1e9),
                         gpu_footprint=24 * int(1e9), exact=False, nlists=None, faiss_M=None,
                         reducer='umap')

    sim.add(x)

    sim.to_path('/tmp/news-sim-cpu')


def nlp_example():
    from src.beam.similarity import TFIDF, SparnnSimilarity, DenseSimilarity, TextSimilarity
    from sklearn.datasets import fetch_20newsgroups

    logger.info(f"Loaded dataset: newsgroups_train")
    newsgroups_train = fetch_20newsgroups(subset='train')
    # newsgroups_test = fetch_20newsgroups(subset='test')

    text_sim = TextSimilarity()
    text_sim.add(newsgroups_train.data[:100])
    print(text_sim.nlp("Find the two closest sentences to the sentence \"it is a beautiful day\"", llm='openai:///gpt-4'))


def sparnn_example():
    from scipy.sparse import csr_matrix
    from src.beam.similarity import SparnnSimilarity

    features = np.random.binomial(1, 0.01, size=(1000, 20000))
    features = csr_matrix(features)

    # build the search index!
    data_to_return = range(1000)

    sim = SparnnSimilarity()
    sim.fit(features, index=data_to_return)
    print(sim.search(features[:5], k=3))


def get_name():
    from beam.core import Processor

    abcd = Processor()

    print(abcd.name)


def beam_data_slice():

    from src.beam import BeamData
    # bd = BeamData(['hi how are you?', 'I am fine, thank you very much', 'the yellow submarine is here'],
    #               index=['a', 'b', 'c'])

    # print(bd[['a']])

    bd = BeamData(['hi how are you?', 'I am fine, thank you very much', 'the yellow submarine is here'])
    print(bd[[0, 0, 1]])


def load_algorithm():
    path = '/dsi/shared/elads/elads/data/tabular/results/deep_tabular/debug_reporter/covtype/0000_20240111_200041'
    from src.beam.tabular import DeepTabularAlg
    alg = DeepTabularAlg.from_pretrained(path)
    print(alg)


def load_model():

    from src.beam.auto import AutoBeam
    alg = AutoBeam.from_bundle('/tmp/mnist_bundle')
    print(alg)


def test_beam_parallel():

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


def build_hparams():
    from src.beam.tabular import TabularConfig
    hparams = TabularConfig(data_path='xxxxx')

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

    from src.beam.tabular import TabularTransformer, TabularConfig, DeepTabularAlg

    kwargs_base = dict(algorithm='debug_reporter',
                       # data_path='/dsi/shared/elads/elads/data/tabular/dataset/data/',
                       data_path='/home/dsi/elads/data/tabular/data/',
                       logs_path='/dsi/shared/elads/elads/data/tabular/results/',
                       copy_code=False, dynamic_masking=False, comet=False, tensorboard=True, n_epochs=2,
                       stop_at=0.98, n_gpus=1, device=1, n_quantiles=6, label_smoothing=.2)

    kwargs_all = {}

    k = 'california_housing'
    kwargs_all[k] = dict(batch_size=128)

    logger.info(f"Starting a new experiment with dataset: {k}")
    hparams = {**kwargs_base}
    hparams.update(kwargs_all[k])
    hparams['dataset_name'] = k
    hparams['identifier'] = k
    hparams = TabularConfig(hparams)

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
        "--n-epochs=50 --epoch-length-train=50000 --epoch-length-eval=10000 --clip=0 --n-gpus=1 --accumulate=1 --no-deterministic",
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
    alg = AutoBeam.from_bundle(path)
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
    from beam.serve.client import BeamClient

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

    # parallel_treading()

    # load_model()

    # load_algorithm()

    # beam_data_slice()

    # get_name()

    # sparnn_example()

    # nlp_example()

    # save_index()

    # load_index()

    # sftp_example()

    # distributed_server()

    # distributed_client()

    # grpc_server()

    # simple_server()

    # test_beam_data_keys()

    # test_beam_parallel()

    # test_mlflow_path()

    # test_transformer()

    # test_config()

    # test_recursive_len()

    # test_slice_to_index()

    # test_catboost()

    # test_collate_transformer_chunks()

    # test_special_attributes()

    test_beam_default_configuration()

    print('done')