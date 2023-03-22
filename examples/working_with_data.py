
from examples.example_utils import add_beam_to_path
add_beam_to_path()

from src.beam.data import BeamData
from src.beam.path import beam_path
from src.beam.processor import Transformer
import numpy as np
import pandas as pd
import torch

import string
import random


def rand_column(n=4):
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for i in range(n))


class MyTransformer(Transformer):

    def __init__(self, *args, n_workers=0, n_chunks=None, chunksize=None, **kwargs):
        super().__init__(*args, n_workers=n_workers, n_chunks=n_chunks, chunksize=chunksize, **kwargs)

    def transform_callback(self, x, key=None, is_chunk=True, **kwargs):
        return len(x)


def get_data(form=None, n=100, m=100, k=4):

    if form == 1:

        data = {'a': {rand_column(): pd.DataFrame(index=np.random.permutation(np.arange(n)),
                                                 data=np.random.randn(n, m), columns=[rand_column() for _ in range(m)]) for
                     _ in range(k)},
               'b': [pd.DataFrame(index=np.random.permutation(np.arange(n)),
                                  data=np.random.randn(n, m), columns=[rand_column() for _ in range(m)]) for _ in range(k)]}

    elif form == 2:

        data = {'a': {rand_column(): pd.DataFrame(index=np.random.permutation(np.arange(n)),
                                                  data=np.random.randn(n, m), columns=[rand_column() for _ in range(m)])
                      for
                      _ in range(k)},
                'b': [pd.DataFrame(index=np.random.permutation(np.arange(n)),
                                   data=np.random.randn(n, m), columns=[rand_column() for _ in range(m)]) for _ in
                      range(k)]}

        data['a']['x'] = pd.DataFrame(data=np.random.permutation(np.arange(n)))

    elif form == 3:

        data = {'a': torch.randn(200, m), 'b': torch.randn(300, m), 'c': torch.randn(400, m)}

    elif form == 4:

        data = pd.DataFrame(index=np.random.permutation(np.arange(n)),
                            data=np.random.randn(n, m), columns=[rand_column() for _ in range(m)])

    else:
        data = pd.DataFrame(index=np.random.permutation(np.arange(n)))

    return data


def get_path(path, storage='local'):

    access_key = 'EBemHypH7I2NcHx1'
    secret_key = 'cvYL26ItASAwE8ZUxRaZKhVVdcxHZ0SJ'

    if storage == 's3':
        path = beam_path(f"s3://192.168.10.45:9000{path}", access_key=access_key, secret_key=secret_key, tls=False)
    elif storage == 'sftp':
        path = beam_path(f'sftp://elads@dsigpu04/dsi/shared/elads/elads/{path}')

    return beam_path(path)


if __name__ == '__main__':

    # tests = ['transform_dd']
    # tests = ['single_file']
    # tests = ['chunks']
    # tests = ['store_and_reload']
    # tests = ['load_data']
    tests = ['empty_path']
    storage = 'file'
    # storage = 'sftp'

    if 'empty_path' in tests:
        print('starting empty_path')

        bd = BeamData.from_path('/tmp/sandbox/yy')
        print(bd)
        bd.cache()
        print(bd)
        print('done empty_path')

    if 'loc_ops' in tests:

        print("starting loc ops")

        path = get_path('/tmp/sandbox/bd', storage)

        bd = BeamData(path=path)
        bd.cache()

        print(bd.iloc[0:2])

        # iloc/loc operations for index type
        bd = BeamData(data=get_data(form=3))
        print(bd.iloc[0:2])

        print("done loc ops")

    if 'load_data' in tests:

        print("starting load data")

        path = get_path('/tmp/sandbox/bd', storage)

        data = get_data(1)

        bd = BeamData(data, path=path, archive_size=0)
        bd.store()

        bd = BeamData(path=path)
        print(bd)
        print(bd.values)

        print("done load data")


    if 'set_item' in tests:

        print("starting set item")

        path = get_path('/tmp/sandbox/bd', storage)
        data = get_data(1)

        bd = BeamData(data, path=path, archive_size=0)
        bd.store()

        bd = BeamData.from_path(path=path)
        bd['a', 'x'] = pd.DataFrame(data=np.random.permutation(np.arange(100)))

        bd.store()
        print("done set item")

    if 'store_and_reload' in tests:

        print("starting store and reload")

        path = get_path('/tmp/sandbox/xx', storage)
        data = get_data(4)

        xx = BeamData(data, path=path)
        xx.store()

        yy = BeamData.from_path(path)

        print(yy)
        yy.cache()

        print("done store and reload")

    if 'data_ops' in tests:

        print("starting data ops")

        path = get_path('/tmp/sandbox/bd', storage)
        data = get_data(1)

        bd = BeamData(data, path=path)

        print(bd.key['a', 'x'])

        q = bd[:10]

        l = list(iter(bd))

        print(bd.stack)

        bd.store()

        be = BeamData.from_path(path)
        be.cache()

        print(bd.orientation)

        keys = list(bd.keys().keys())
        bda = bd[keys[0]]

        info = bda.info
        # bda = bd['a']
        bd.store()

        print("done data ops")

    if 'transform_dd' in tests:

        print("starting transform")

        # path = get_path('/tmp/sandbox/bd', storage)
        # data = get_data(2)
        #
        # bd = BeamData(data, path=path)
        # bd.store()
        #
        # bd2 = BeamData(path=path)
        # bd2.cache()

        data = get_data(form=2, n=100, m=100, k=40)
        bd2 = BeamData(data)

        tf = MyTransformer(n_chunks=10, n_workers=0)
        y = tf.transform(bd2)

        print("done transform_dd")

    if 'transform_df' in tests:
        print("starting transform")

        path = get_path('/tmp/sandbox/bd', storage)
        data = get_data(2)

        bd = BeamData(data, path=path)
        bd.store()

        bd2 = BeamData(path=path)
        bd2.cache()

        tf = MyTransformer(n_chunks=2, n_workers=0)
        y = tf.transform(bd2)

        print("done transform")

    if 'single_file' in tests:

            print("starting single file")

            path = get_path('/tmp/sandbox/bd', storage)
            data = get_data(1)

            bd = BeamData(data, path=path, archive_size=0)
            bd.store()

            path2 = list(bd.all_paths['a'].values())[0]

            bd2 = BeamData(path=path2)
            bd2.cache()
            print(bd2)

            bd = BeamData(path=path)
            bd.cache()
            print(bd)

            bd3 = BeamData(path=path, all_paths={'data': path2})
            bd3.cache()
            print(bd3)

            print("done single file")

    if 'chunks' in tests:

            print("starting chunks")

            path = get_path('/tmp/sandbox/bd', storage)
            data = get_data(1)

            bd = BeamData(data, path=path, archive_size=0, n_chunks=2, split_by='index')
            bd.store()

            bd2 = BeamData(path=path)
            bd2.cache()

            print(bd2)

            print("done chunks")
    if 'iter' in tests:

            print("starting iter")

            path = get_path('/tmp/sandbox/bd', storage)
            data = get_data(1)

            bd = BeamData(data, path=path, archive_size=0, n_chunks=2, split_by='index')
            bd.store()

            for k, v in bd:
                print(k, v)

            print("done iter")


    print("done all tests")
