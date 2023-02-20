from .utils import divide_chunks, collate_chunks, recursive_chunks, iter_container, logger, \
    recursive_size_summary, container_len, is_arange, listdir_fullpath, retrieve_name
from .parallel import parallel, BeamParallel, BeamTask
from collections import OrderedDict
from .data import BeamData
from .path import beam_path


class Processor(object):

    def __init__(self, *args, name=None, **kwargs):
        self._name = name

    @property
    def name(self):
        if self._name is None:
            self._name = retrieve_name(self)
        return self._name


class Pipeline(Processor):

    def __init__(self, *ts, track_steps=False, name=None, **kwts):

        super().__init__(name=name)
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

    def __init__(self, *args, state=None, n_workers=0, n_chunks=None, name=None,
                 chunksize=None, squeeze=True, path=None, multiprocess_method='joblib',
                 reduce_dim=0, **kwargs):

        super(Transformer, self).__init__(*args, name=name, **kwargs)

        if (n_chunks is None) and (chunksize is None):
            n_chunks = 1

        self.transformers = None
        self.chunksize = chunksize
        self.n_chunks = n_chunks
        self.n_workers = n_workers
        self.state = state
        self.squeeze = squeeze
        self.kwargs = kwargs

        if path is not None:
            path = beam_path(path)
        if path is not None and name is not None:
            path = path.joinpath(name)

        self.path = path
        self.multiprocess_method = multiprocess_method
        self.reduce_dim = reduce_dim

        self.queue = BeamParallel(n_workers=n_workers, func=None, method=multiprocess_method,
                                  progressbar='beam', reduce=False, reduce_dim=reduce_dim, **kwargs)

    def chunks(self, x, chunksize=None, n_chunks=None, squeeze=None):

        if (chunksize is None) and (n_chunks is None):
            chunksize = self.chunksize
            n_chunks = self.n_chunks
        if squeeze is None:
            squeeze = self.squeeze

        if isinstance(x, BeamData) and x.cached and x.orientation == 'simple':

            data_chunks = recursive_chunks(x.data, chunksize=chunksize, n_chunks=n_chunks, squeeze=squeeze)
            index_chunks = recursive_chunks(x.index, chunksize=chunksize, n_chunks=n_chunks, squeeze=squeeze)
            label_chunks = recursive_chunks(x.label, chunksize=chunksize, n_chunks=n_chunks, squeeze=squeeze)

            for dc, ic, lc in zip(data_chunks, index_chunks, label_chunks):

                k = dc[0]
                c = x.clone(data=dc[1], index=ic[1], label=lc[1], columns=x.columns)

                yield k, c

        else:
            for k, c in recursive_chunks(x, chunksize=chunksize, n_chunks=n_chunks, squeeze=squeeze):
                yield k, c

    def transform_callback(self, x, key=None, is_chunk=False, fit=False, path=None, **kwargs):
        raise NotImplementedError

    def worker(self, x, key=None, is_chunk=False, fit=False, cache=True, store_path=None, **kwargs):

        if isinstance(x, BeamData):
            if not x.cached and cache:
                x.cache()

        x = self.transform_callback(x, key=key, is_chunk=is_chunk, fit=fit, **kwargs)

        if isinstance(x, BeamData):
            if store_path is not None:
                x.store(path=store_path)
                x = BeamData.from_path(path=store_path)

        return x

    def fit(self, x, **kwargs):
        return x

    def fit_transform(self, x, **kwargs):
        return self.transform(x, fit=True, **kwargs)
        # self.fit(x, **kwargs)
        # return self.transform(x, **kwargs)

    def collate(self, x, reduce_dim=None, **kwargs):

        if reduce_dim is None:
            reduce_dim = self.reduce_dim

        return collate_chunks(*x, dim=reduce_dim, **kwargs)

    def transform(self, x, chunksize=None, n_chunks=None, n_workers=None, squeeze=None, multiprocess_method=None,
                  fit=False, cache=True, store=False, store_chunk=False, path=None, **kwargs):

        if path is None:
            path = self.path

        logger.info(f"Starting transformer process: {self.name}")

        if len(x) == 0:
            return x

        if (chunksize is None) and (n_chunks is None):
            chunksize = self.chunksize
            n_chunks = self.n_chunks
        if squeeze is None:
            squeeze = self.squeeze

        is_chunk = (n_chunks != 1) or (not squeeze)
        self.queue.set_name(self.name)

        if is_chunk:
            for k, c in self.chunks(x, chunksize=chunksize, n_chunks=n_chunks, squeeze=squeeze):

                chunk_path = None
                if store_chunk and path is not None:
                    chunk_path = path.joinpath(beam_path(path), BeamData.normalize_key(k))

                self.queue.add(BeamTask(self.worker, (c, ), {'key': k, 'is_chunk': is_chunk,
                                                             'cache': cache, fit: fit, 'path': chunk_path,
                                                             'store': store_chunk},
                                        name=f"{self.name}/{k}", **kwargs))

        else:
            self.queue.add(BeamTask(self.worker, (x, ), {'key': None, 'is_chunk': is_chunk,
                                                         'cache': cache, fit: fit},
                                    name=f"{self.name}", **kwargs))

        x = self.queue.run(n_workers=n_workers, method=multiprocess_method)

        logger.info(f"Finished transformer process: {self.name}. Collating results...")

        if isinstance(x[0], BeamData):
            x = BeamData.collate(x)
            if store and path is not None:
                x.store(path=path)
                x = BeamData.from_path(path=path)
        else:
            x = self.collate(x, **kwargs)

        return x





