from .utils import divide_chunks, collate_chunks, recursive_chunks, iter_container, logger, \
    recursive_size_summary, container_len, is_arange, listdir_fullpath
from .parallel import parallel, task
from collections import OrderedDict
from .data import BeamData


class Processor(object):

    def __init__(self, *args, **kwargs):
        pass



class Pipeline(Processor):

    def __init__(self, *ts, track_steps=False, **kwts):

        super().__init__()
        self.track_steps = track_steps
        self.steps = {}

        self.transformers = OrderedDict()
        for i, t in enumerate(ts):
            self.transformers[i] = t

        for k, t in kwts.items():
            self.transformers[k] = t

    def transform(self, x, **kwargs):

        self.steps = []

        for i, t in self.transformers.items():

            kwargs_i = kwargs[i] if i in kwargs.keys() else {}
            x = t.transform(x, **kwargs_i)

            if self.track_steps:
                self.steps[i] = x

        return x


class Reducer(Processor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reduce(self, *xs, **kwargs):
        return collate_chunks(*xs, dim=1, **kwargs)


class Transformer(Processor):

    def __init__(self, *args, state=None, n_jobs=0, n_chunks=None,
                 chunksize=None, squeeze=True, path=None, **kwargs):

        super(Transformer, self).__init__(*args, **kwargs)

        if (n_chunks is None) and (chunksize is None):
            n_chunks = 1

        self.transformers = None
        self.chunksize = chunksize
        self.n_chunks = n_chunks
        self.n_jobs = n_jobs
        self.state = state
        self.squeeze = squeeze
        self.kwargs = kwargs
        self.path = path

    def chunks(self, x, chunksize=None, n_chunks=None, squeeze=None):

        if (chunksize is None) and (n_chunks is None):
            chunksize = self.chunksize
            n_chunks = self.n_chunks
        if squeeze is None:
            squeeze = self.squeeze

        for k, c in recursive_chunks(x, chunksize=chunksize, n_chunks=n_chunks, squeeze=squeeze):
            yield k, c

    def transform_callback(self, x, key=None, is_chunk=False, fit=False, **kwargs):
        raise NotImplementedError

    def worker(self, x, key=None, is_chunk=False, data_in='memory', strategy='memory', fit=False, **kwargs):

        if data_in == 'disk':
            bd = BeamData(data_paths=x)
            bd.to_memory()
            x = bd.data

        x = self.transform_callback(x, key=key, is_chunk=is_chunk, **kwargs)

        if strategy == 'disk':
            bd = BeamData(x, path=self.path)
            bd.to_disk()
            x = bd.data_paths

        return x

    def fit(self, x, **kwargs):
        return NotImplementedError

    def fit_transform(self, x, **kwargs):
        self.fit(x, **kwargs)
        return self.transform(x, **kwargs)

    def collate(self, x, **kwargs):
        return collate_chunks(*x, dim=0, **kwargs)

    def transform(self, x, chunksize=None, n_chunks=None, n_jobs=None, squeeze=None, parent_strategy='disk',
                  worker_strategy='memory', **kwargs):

        if (chunksize is None) and (n_chunks is None):
            chunksize = self.chunksize
            n_chunks = self.n_chunks
        if squeeze is None:
            squeeze = self.squeeze
        if n_jobs is None:
            n_jobs = self.n_jobs

        data_in = 'memory'
        if isinstance(x, BeamData):
            if parent_strategy == 'disk':
                if x.in_disk:
                    x = x.data_paths
                else:
                    x.to_disk()
                    x = x.data_paths
                data_in = 'disk'

            elif parent_strategy == 'memory':
                if x.in_memory:
                    x = x.data
                else:
                    x.to_memory()
                    x = x.data

        chunks = []
        kwargs_list = []

        is_chunk = (chunksize != 1) or (not squeeze)

        for k, c in self.chunks(x, chunksize=chunksize, n_chunks=n_chunks, squeeze=squeeze):
            chunks.append((c, ))
            kwargs_list.append({'key': k, 'is_chunk': is_chunk,
                                'strategy': worker_strategy, 'data_in': data_in})

        x = parallel(self.worker, chunks, constant_kwargs=kwargs, kwargs_list=kwargs_list,
                           workers=n_jobs, method='apply_async', collate=False)

        return self.collate(x, **kwargs)





