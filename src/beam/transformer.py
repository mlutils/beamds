from .utils import divide_chunks, collate_chunks, recursive_chunks, iter_container, \
    recursive_size_summary, container_len, retrieve_name, build_container_from_tupled_keys, is_empty
from .parallel import parallel, BeamParallel, BeamTask
from collections import OrderedDict
from .data import BeamData
from .path import beam_path
import pickle
import io
from .utils import tqdm_beam as tqdm
from .logger import beam_logger as logger
from .processor import Processor


class Transformer(Processor):

    def __init__(self, *args, n_workers=0, n_chunks=None, name=None, store_path=None, partition=None,
                 chunksize=None, multiprocess_method='joblib', squeeze=True, reduce=True, reduce_dim=0,
                 transform_strategy=None, split_by='keys', **kwargs):
        """

        @param args:
        @param n_workers:
        @param n_chunks:
        @param name:
        @param store_path:
        @param chunksize:
        @param multiprocess_method:
        @param squeeze:
        @param reduce_dim:
        @param transform_strategy: Determines the strategy of cache/store operations during transformation:
            'CC' - the data is cached before the split into multiple chunks and the split to multiprocess,
            the output of each process remains cached and is returned to the main process as a list of cached data.
            'CS' - the data is cached before the split into multiple chunks and the split to multiprocess,
            the output of each process is stored and is returned to the main process as a list of paths.
            This approach suits for enriching the data with additional information, e.g. embeddings
            where the transformed data does not fit into the memory.
            'SC' - the data stored and given to the transformer as a list of paths, the output of each process remains
            cached and is returned to the main process as a list of cached data. This approach suits for the case
            when the input data is too large to fit into the memory but the transformation generate a small output
            that can be cached, e.g. aggregation operations.
            'SS' - the data stored and given to the transformer as a list of paths, the output of each process is stored
            and is returned to the main process as a list of paths. This approach suits for the case when the input data
            is too large to fit into the memory and the transformation generate a large output that cannot be cached,
            e.g. image transformations.
            'C' - the input type is inferred from the BeamData object and the output is cached.
            'S' - the input type is inferred from the BeamData object and the output is stored.
        @param split_by: The split strategy of the data into chunks.
        'keys' - the data is split by the key,
        'index' - the data is split by the index (i.e. dim=0).
        'columns' - the data is split by the columns (i.e. dim=1).
        @param kwargs:
        """
        super(Transformer, self).__init__(*args, name=name, **kwargs)

        if (n_chunks is None) and (chunksize is None):
            n_chunks = 1

        self.transformers = None
        self.chunksize = chunksize
        self.n_chunks = n_chunks
        self.n_workers = n_workers
        self.squeeze = squeeze
        self.kwargs = kwargs
        self.transform_strategy = transform_strategy
        self.split_by = split_by
        if self.transform_strategy in ['SC', 'SS'] and self.split_by != 'keys':
            logger.warning(f'transformation strategy {self.transform_strategy} supports only split_by=\"keys\", '
                           f'The split_by is set to "keys".')
            self.split_by = 'keys'

        if store_path is not None:
            store_path = beam_path(store_path)
        if store_path is not None and name is not None:
            store_path = store_path.joinpath(name)

        self.store_path = store_path
        self.partition = partition
        self.multiprocess_method = multiprocess_method
        self.reduce_dim = reduce_dim
        self.to_reduce = reduce
        self._exceptions = None

    def chunks(self, x, chunksize=None, n_chunks=None, squeeze=None, split_by=None, partition=None):

        if split_by is None:
            split_by = self.split_by

        if partition is None:
            partition = self.partition

        if (chunksize is None) and (n_chunks is None):
            chunksize = self.chunksize
            n_chunks = self.n_chunks
        if squeeze is None:
            squeeze = self.squeeze

        if isinstance(x, BeamData):
            for k, c in x.divide_chunks(chunksize=chunksize, n_chunks=n_chunks, partition=partition, split_by=split_by):
                yield k, c

        else:

            dim = 0 if split_by == 'index' else 1 if split_by == 'column' else None
            for k, c in recursive_chunks(x, chunksize=chunksize, n_chunks=n_chunks, squeeze=squeeze, dim=dim):
                yield k, c

    def transform_callback(self, x, key=None, is_chunk=False, fit=False, path=None, **kwargs):
        raise NotImplementedError

    def worker(self, x, key=None, is_chunk=False, fit=False, cache=True, store_path=None, **kwargs):

        if isinstance(x, BeamData):
            if not x.is_cached and cache:
                x.cache()

        x = self.transform_callback(x, key=key, is_chunk=is_chunk, fit=fit, **kwargs)

        if store_path is not None:
            store_path = beam_path(store_path)
            if not isinstance(x, BeamData):
                x = BeamData(x)
            x.store(path=store_path,)
            x = BeamData.from_path(path=store_path)

        return key, x

    def fit(self, x, **kwargs):
        return x

    @property
    def exceptions(self):
        return self._exceptions

    @exceptions.setter
    def exceptions(self, exceptions):
        self._exceptions = exceptions

    def fit_transform(self, x, **kwargs):
        return self.transform(x, fit=True, **kwargs)
        # self.fit(x, **kwargs)
        # return self.transform(x, **kwargs)

    def reduce(self, x, reduce_dim=None, split_by=None , **kwargs):

        if isinstance(next(iter_container(x))[1], BeamData):
            x = BeamData.collate(x, split_by=split_by, **kwargs)
        else:

            if reduce_dim is None:
                reduce_dim = self.reduce_dim

            x = collate_chunks(*x, dim=reduce_dim, **kwargs)

        return x

    def transform(self, x, chunksize=None, n_chunks=None, n_workers=None, squeeze=None, multiprocess_method=None,
                  fit=False, path=None, split_by=None, partition=None, transform_strategy=None, cache=True, store=False,
                  reduce=True, **kwargs):
        """

        @param x:
        @param chunksize:
        @param n_chunks:
        @param n_workers:
        @param squeeze:
        @param multiprocess_method:
        @param fit:
        @param path:
        @param split_by:
        @param partition:
        @param transform_strategy: see BeamTransformer.transform_strategy
        @param cache: default True, cache the data before transformation if it is not cached.
        @param store: default False, store the data after transformation if it is not stored.
        @param kwargs:
        @return:
        """
        if split_by is None:
            split_by = self.split_by

        if partition is None:
            partition = self.partition

        if multiprocess_method is None:
            multiprocess_method = self.multiprocess_method

        if n_workers is None:
            n_workers = self.n_workers

        if transform_strategy is None:
            transform_strategy = self.transform_strategy

        if reduce is None:
            reduce = self.to_reduce

        reduce_dim = self.reduce_dim

        if transform_strategy in ['SC', 'SS'] and split_by != 'keys':
            logger.warning(f'transformation strategy {transform_strategy} supports only split_by=\"key\", '
                           f'The split_by is set to "key".')
            split_by = 'keys'

        if path is None:
            path = self.store_path

        logger.info(f"Starting transformer process: {self.name}")

        if is_empty(x):
            return x

        if (chunksize is None) and (n_chunks is None):
            chunksize = self.chunksize
            n_chunks = self.n_chunks
        if squeeze is None:
            squeeze = self.squeeze

        is_chunk = (n_chunks != 1) or (not squeeze) or (split_by == 'keys' and isinstance(x, BeamData) and x.is_stored)

        if ((transform_strategy is None) or (transform_strategy == 'C')) and type(x) == BeamData:
            if x.is_cached:
                transform_strategy = 'CC'
            elif x.is_stored:
                transform_strategy = 'SC'
            else:
                raise ValueError(f"BeamData is not cached or stored, check your configuration")

        if transform_strategy == 'S' and type(x) == BeamData:
            if x.is_cached:
                transform_strategy = 'CS'
            elif x.is_stored:
                transform_strategy = 'SS'
            else:
                raise ValueError(f"BeamData is not cached or stored, check your configuration")

        if transform_strategy in ['CC', 'CS'] and type(x) == BeamData and not x.is_cached:
            logger.warning(f"Data is not cached but the transformation strategy is {transform_strategy}, "
                           f"caching data for transformer: {self.name} before the split to chunks.")
            x.cache()

        if transform_strategy in ['SC', 'SS'] and type(x) == BeamData and not x.is_stored:
            logger.warning(f"Data is not stored but the transformation strategy is {transform_strategy}, "
                           f"storing data for transformer: {self.name} before the split to chunks.")
            x.store()

        store_chunk = transform_strategy in ['CS', 'SS']

        if path is None and store_chunk:

            if isinstance(x, BeamData) and x.path is not None:
                path = x.path
                path = path.parent.joinpath(f"{path.name}_transformed_{self.name}")
                logger.info(f"Path is not specified for transformer: {self.name}, "
                            f"the chunk will be stored in a neighboring directory as the original data: {x.path}"
                            f"to: {path}.")
            else:
                logger.warning(f"Path is not specified for transformer: {self.name}, "
                               f"the chunk will not be stored.")
                store_chunk = False
        elif store_chunk:
            logger.info(f"Storing transformed chunks of data in: {path}")

        queue = BeamParallel(n_workers=n_workers, func=None, method=multiprocess_method, name=self.name,
                                  progressbar='beam', reduce=False, reduce_dim=reduce_dim, **kwargs)

        if is_chunk:
            logger.info(f"Splitting data to chunks for transformer: {self.name}")
            for k, c in tqdm(self.chunks(x, chunksize=chunksize, n_chunks=n_chunks,
                                    squeeze=squeeze, split_by=split_by, partition=partition)):

                chunk_path = None
                if store_chunk:

                    if split_by == 'index':
                        part_name = BeamData.index_partition_directory_name
                    elif split_by == 'columns':
                        part_name = BeamData.columns_partition_directory_name
                    else:
                        part_name = ''

                    chunk_path = path.joinpath(f"{BeamData.normalize_key(k)}{part_name}")
                    # chunk_path = chunk_path.as_uri()

                queue.add(BeamTask(self.worker, c, key=k, is_chunk=is_chunk, fit=fit, store_path=chunk_path,
                                        cache=cache, store=store_chunk, name=f"{self.name}/{k}", **kwargs))

        else:
            queue.add(BeamTask(self.worker, x, key=None, is_chunk=is_chunk, fit=fit, cache=cache,
                                    store=store_chunk,  name=f"{self.name}", **kwargs))

        logger.info(f"Starting transformer: {self.name} with {n_workers} workers. "
                    f"Number of queued tasks is {len(queue)}.")

        x_with_keys = queue.run(n_workers=n_workers, method=multiprocess_method)

        exceptions = [{'exception': xi, 'task': queue.queue[i]} for i, xi in enumerate(x_with_keys)
                      if isinstance(xi, Exception)]

        if len(exceptions) > 0:
            logger.error(f"Transformer {self.name} had {len(exceptions)} exceptions during operation.")
            logger.info("Failed tasks can be obtained in self.exceptions")
            self.exceptions = exceptions

        x_with_keys = [xi for xi in x_with_keys if not isinstance(xi, Exception)]

        if is_chunk:
            values = [xi[1] for xi in x_with_keys]
            keys = [xi[0] for xi in x_with_keys]
            keys = [ki if type(ki) is tuple else (ki, ) for ki in keys]

            if len(exceptions) == 0:
                x = build_container_from_tupled_keys(keys, values)

                logger.info(f"Finished transformer process: {self.name}. Collating results...")

                if reduce:
                    x = self.reduce(x, split_by=split_by, **kwargs)
            else:
                x = {k: v for k, v in zip(keys, values)}
                if store:
                    logger.warning("Due to exceptions, the data will not be stored, "
                                   "the data is returned as a dictionary of all the successful tasks.")
                return x

        else:
            if len(exceptions) > 0:
                logger.warning("Exception occurred, the exception object and the task are returned.")
                return x_with_keys
            logger.info(f"Finished transformer process: {self.name}.")
            x = x_with_keys[0][1]

        if store:

            logger.info(f"Storing transformed of data in: {path}")
            if not isinstance(x, BeamData):
                x = BeamData(x)
            x.store(path=path)
            x = BeamData.from_path(path=path)

        return x


class Reducer(Processor):

    def __init__(self, hparams, *args, dim=1, **kwargs):
        super().__init__(hparams, *args, **kwargs)
        self.dim = dim

    def reduce(self, *xs, **kwargs):
        return collate_chunks(*xs, dim=self.dim, **kwargs)