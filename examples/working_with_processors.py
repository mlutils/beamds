
from examples.example_utils import add_beam_to_path
add_beam_to_path()

from src.beam.processor import BeamData, Transformer
import numpy as np
import pandas as pd

import string
import random

def rand_column(n=16):
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for i in range(n))


class MyTransformer(Transformer):

    def __init__(self, *args, n_jobs=0, n_chunks=None, chunksize=None, **kwargs):
        super().__init__(*args, n_jobs=n_jobs, n_chunks=n_chunks, chunksize=chunksize, **kwargs)

    def _transform(self, x, key=None, is_chunk=True, **kwargs):
        print('xxx')
        return len(x)



if __name__ == '__main__':

    n = 100
    m = 100
    k = 4

    dfs = {'a': {rand_column(): pd.DataFrame(index=np.random.permutation(np.arange(n)),
                        data=np.random.randn(n, m), columns=[rand_column() for _ in range(m)]) for _ in range(k)},
           'b': [pd.DataFrame(index=np.random.permutation(np.arange(n)),
                        data=np.random.randn(n, m), columns=[rand_column() for _ in range(m)]) for _ in range(k)]}

    # dfs = pd.DataFrame(index=np.random.permutation(np.arange(n)), data=np.random.randn(n, m),
    #                    columns=[rand_column() for _ in range(m)])

    # dfs = [1, 2, 3]

    path = '/tmp/sandbox/multi_dfs'
    bd = BeamData(dfs, path=path)
    # bd.write(dfs)

    bd.write(dfs, n_chunks=2)

    x = bd.read()

    bd2 = BeamData(path=path)

    bd2.to_memory()

    # tf = MyTransformer(chunksize=1, n_jobs=0)
    tf = MyTransformer(n_chunks=2, n_jobs=0)
    y = tf.transform(bd2, parent_strategy='disk', worker_strategy='disk')

    print("done")