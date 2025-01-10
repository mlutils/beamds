import inspect
from enum import Enum

from ..utils import (collate_chunks, recursive_chunks, iter_container,
                     build_container_from_tupled_keys, is_empty, check_type, retry, get_number_of_cores)
from ..concurrent import BeamParallel, BeamTask
from ..data import BeamData
from ..path import beam_path
from ..utils import tqdm_beam as tqdm
from ..logging import beam_logger as logger
from ..processor.core import Processor
from ..base import beam_cache
from ..config import TransformerConfig
from ..type import is_beam_data, Types


class TransformStrategy(Enum):
    CC = "CC"
    CS = "CS"
    SC = "SC"
    SS = "SS"
    C = "C"
    S = "S"


class Transformer(Processor):

    def __init__(self, *args, func=None, n_workers=0, n_chunks=None, name=None, store_path=None, partition=None,
                 chunksize=None, mp_method='joblib', squeeze=False, reduce=True, reduce_dim=0, store_chunk=None,
                 transform_strategy=None, split_by='keys', store_suffix=None, shuffle=False, override=False,
                 use_dill=False, return_results=None, use_cache=False, retries=1, silent=False, reduce_func=None,
                 chunksize_policy='round', retries_delay=1., _config_scheme=None, strict=False, **kwargs):
        """

        @param args:
        @param n_workers:
        @param n_chunks:
        @param name:
        @param store_path:
        @param chunksize:
        @param mp_method:
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
        @param store_suffix: The suffix of the stored file.
        @param shuffle Shuffling the tasks before running them.
        @param kwargs:
        """
        assert inspect.isroutine(func) or func is None, "The func argument must be a function."

        name = name or func.__name__ if func is not None else None
        _config_scheme = _config_scheme or TransformerConfig
        super(Transformer, self).__init__(*args, name=name, n_workers=n_workers, n_chunks=n_chunks,
                                          store_path=store_path, partition=partition, chunksize=chunksize,
                                          mp_method=mp_method, squeeze=squeeze, reduce=reduce, reduce_dim=reduce_dim,
                                          store_chunk=store_chunk, transform_strategy=transform_strategy,
                                          split_by=split_by, store_suffix=store_suffix, shuffle=shuffle,
                                          return_results=return_results, reduce_func=reduce_func,
                                          override=override, use_dill=use_dill, use_cache=use_cache,
                                          _config_scheme=_config_scheme, retries=retries, silent=silent,
                                          retries_delay=retries_delay, chunksize_policy=chunksize_policy, strict=strict,
                                          **kwargs)

        self.func = func
        self.reduce_func = reduce_func

        # check if we can pass kwargs to the function
        self.func_has_kwargs = False
        if func is not None:
            sig = inspect.signature(func)
            self.func_has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            self.func_has_kwargs = self.func_has_kwargs or 'transformer_kwargs' in sig.parameters

        self.chunksize = self.hparams.chunksize
        self.n_chunks = self.hparams.n_chunks
        if (self.n_chunks is None) and (self.chunksize is None):
            self.n_chunks = 1

        self.n_workers = self.hparams.n_workers
        self.squeeze = self.hparams.squeeze
        self.split_by = self.hparams.split_by
        self.store_suffix = self.hparams.store_suffix
        self.transform_strategy = self.hparams.transform_strategy
        self.store_chunk = self.hparams.store_chunk
        self.shuffle = self.hparams.shuffle
        self.override = self.hparams.override
        self.use_dill = self.hparams.use_dill
        self.return_results = self.hparams.return_results
        self.use_cache = self.hparams.use_cache
        self.retries = self.hparams.retries
        self.retries_delay = self.hparams.retries_delay
        self.silent = self.hparams.silent
        self.chunksize_policy = self.hparams.chunksize_policy
        self.strict = strict or self.hparams.strict_transform

        if self.transform_strategy in [TransformStrategy.SC, TransformStrategy.SS] and self.split_by != 'keys':
            logger.warning(f'transformation strategy {self.transform_strategy} supports only split_by=\"keys\", '
                           f'The split_by is set to "keys".')
            self.split_by = 'keys'

        store_path = store_path or self.get_hparam('store_path')
        if store_path is not None:
            store_path = beam_path(store_path)
        if store_path is not None and name is not None:
            store_path = store_path.joinpath(name)

        self.store_path = store_path
        self.partition = self.hparams.partition
        self.mp_method = self.hparams.mp_method
        self.reduce_dim = self.hparams.reduce_dim
        self.to_reduce = self.hparams.reduce
        self._exceptions = None
        self.counter = 0

    def __call__(self, x, **kwargs):
        return self.transform(x, **kwargs)

    def reset(self):
        self.counter = 0

    def chunks(self, x, chunksize=None, n_chunks=None, squeeze=None, split_by=None, partition=None,
               chunksize_policy=None):

        split_by = split_by or self.split_by
        partition = partition or self.partition

        if (chunksize is None) and (n_chunks is None):
            chunksize = self.chunksize
            n_chunks = self.n_chunks
        if squeeze is None:
            squeeze = self.squeeze

        if chunksize_policy is None:
            return self.chunksize_policy

        if is_beam_data(x):
            for k, c in x.divide_chunks(chunksize=chunksize, n_chunks=n_chunks, partition=partition,
                                        split_by=split_by, chunksize_policy=chunksize_policy):
                yield k, c

        else:

            dim = 0 if split_by == 'index' else 1 if split_by == 'column' else None
            for k, c in recursive_chunks(x, chunksize=chunksize, n_chunks=n_chunks, squeeze=squeeze, dim=dim,
                                         chunksize_policy=chunksize_policy):
                yield k, c

    def transform_callback(self, x, _key=None, _is_chunk=False, _fit=False, path=None, _store=False, **kwargs):

        if self.func is None:
            raise ValueError("The function is not defined for the transformer. Either pass fanc to the constructor or "
                             "override the transform_callback method.")

        if self.func_has_kwargs:
            kwargs['transformer_kwargs'] = dict(key=_key, is_chunk=_is_chunk, fit=_fit, path=path, store=_store)
        r = self.func(x, **kwargs)

        return r

    def worker(self, x, key=None, is_chunk=False, fit=False, cache=True, store_path=None, store=False,
               override=False, return_results=True, use_cache=False, task_kwargs=None, retries=1, retries_delay=1.0,):

        task_kwargs = task_kwargs or {}

        if isinstance(x, BeamData):
            if not x.is_cached and cache:
                x.cache()

        if store_path is not None:
            store_path = beam_path(store_path)
            if BeamData.exists(store_path):
                if use_cache:
                    logger.debug(f"File/path {store_path} exists using it as cache and skipping calculation.")
                    if return_results:
                        return key, BeamData.read(store_path)
                    else:
                        return key, None

                elif override:
                    logger.warning(f"File/path {store_path} exists, the data will be stored with the same name "
                                   f"(override=True).")
                    store_path.unlink()
                else:
                    logger.warning(f"File/path {store_path} already exists, the data will not be stored (override=False).")
                    if return_results:
                        return key, BeamData.read(store_path)
                    else:
                        return key, None

        if retries > 1:
            transform = retry(self.transform_callback, retries=retries, sleep=retries_delay)
        else:
            transform = self.transform_callback

        x = transform(x, _key=key, _is_chunk=is_chunk, _fit=fit, _store=store, **task_kwargs)

        if store_path is not None:
            store_path = beam_path(store_path)
            if store_path.suffix:
                logger.info(f"Storing transformed chunk in: {store_path}")
                store_path.write(x)
            else:
                if isinstance(x, BeamData):
                    x.store(path=store_path)
                else:
                    BeamData.write_tree(x, path=store_path, split=False)

        if return_results:
            return key, x
        else:
            return key, None

    def fit(self, x, **kwargs):
        raise NotImplementedError("Override the fit method to implement the fitting process.")

    @property
    def exceptions(self):
        return self._exceptions

    @exceptions.setter
    def exceptions(self, exceptions):
        self._exceptions = exceptions

    def fit_transform(self, x, **kwargs):
        return self.transform(x, fit=True, **kwargs)

    def reduce(self, y, reduce_dim=None, split_by=None, squeeze=True, x_in=None):

        if self.reduce_func is not None:
            y = self.reduce_func(y)
        elif isinstance(next(iter_container(y))[1], BeamData):
            y = BeamData.collate(y, split_by=split_by)
        else:

            if reduce_dim is None:
                reduce_dim = self.reduce_dim

            if isinstance(y, list):
                y = collate_chunks(*y, dim=reduce_dim, squeeze=squeeze)
            elif isinstance(y, dict):
                y = collate_chunks(*list(y.values()), keys=list(y.keys()), dim=reduce_dim, squeeze=squeeze,
                                   logger=logger)
            else:
                raise TypeError(f"Unsupported type for reduction: {type(y)} (supports list and dict).")

        return y

    @beam_cache(exception_keys=['parallel_kwargs'])
    def cached_transform(self, x, transform_kwargs=None, parallel_kwargs=None, **kwargs):
        return self.transform(x, transform_kwargs=transform_kwargs, parallel_kwargs=parallel_kwargs, **kwargs)

    def transform(self, x, transform_kwargs=None, parallel_kwargs=None, **kwargs):

        transform_kwargs = transform_kwargs or {}

        split_by = transform_kwargs.pop('split_by', self.split_by)
        partition = transform_kwargs.pop('partition', self.partition)
        shuffle = transform_kwargs.pop('shuffle', self.shuffle)
        store_suffix = transform_kwargs.pop('store_suffix', self.store_suffix)
        transform_strategy = transform_kwargs.pop('transform_strategy', self.transform_strategy)
        store_chunk = transform_kwargs.pop('store_chunk', self.store_chunk)
        reduce = transform_kwargs.pop('reduce', self.to_reduce)
        store_path = beam_path(transform_kwargs.pop('store_path', self.store_path))
        override = transform_kwargs.pop('override', self.override)
        store = transform_kwargs.pop('store', (store_path is not None))
        return_results = transform_kwargs.pop('return_results', self.return_results)
        reduce_dim = transform_kwargs.pop('reduce_dim', self.reduce_dim)
        use_cache = transform_kwargs.pop('use_cache', self.use_cache)
        silent = transform_kwargs.pop('silent', self.silent)
        chunksize_policy = transform_kwargs.pop('chunksize_policy', self.chunksize_policy)
        strict = transform_kwargs.pop('strict', self.strict)

        parallel_kwargs = parallel_kwargs or {}
        n_workers = parallel_kwargs.pop('n_workers', self.n_workers)
        mp_method = parallel_kwargs.pop('mp_method', self.mp_method)
        use_dill = parallel_kwargs.pop('use_dill', self.use_dill)
        retries = parallel_kwargs.pop('retries', self.retries)
        retries_delay = parallel_kwargs.pop('retries_delay', self.retries_delay)

        logger.info(f"Starting transformer process: {self.name}")

        if is_empty(x):
            return x

        chunksize = transform_kwargs.pop('chunksize', self.chunksize)
        n_chunks = transform_kwargs.pop('n_chunks', self.n_chunks)
        squeeze = transform_kwargs.pop('squeeze', self.squeeze)
        if (chunksize is None) and (n_chunks is None):
            chunksize = self.chunksize
            n_chunks = self.n_chunks
        if (chunksize is None) and (n_chunks is None):
            n_chunks = 1

        if n_workers is None:
            if chunksize is not None:
                n_workers = 1
            else:
                n_workers = n_chunks
        elif n_workers < 1:
            # defaults to half of the available cores
            n_workers = get_number_of_cores() // 2

        if squeeze is None:
            squeeze = self.squeeze

        # the default behavior is to store the chunk if the data is chunked and store path is not None
        if store_chunk is None:
            store_chunk = ((n_chunks is not None and n_chunks > 1) or chunksize is not None) and store

        if transform_strategy is None and store_chunk is not None:
            transform_strategy = TransformStrategy.S if store_chunk else TransformStrategy.C

        if transform_strategy in [TransformStrategy.SC, TransformStrategy.SS] and split_by != 'keys':
            logger.warning(f'transformation strategy {transform_strategy} supports only split_by=\"key\", '
                           f'The split_by is set to "key".')
            split_by = 'keys'

        if split_by == 'index':
            part_name = BeamData.index_partition_directory_name
        elif split_by == 'columns':
            part_name = BeamData.columns_partition_directory_name
        else:
            part_name = ''

        is_chunk = (n_chunks != 1) or (not squeeze) or (split_by == 'keys' and isinstance(x, BeamData) and x.is_stored)

        if ((transform_strategy is None) or (transform_strategy == TransformStrategy.C)) and type(x) == BeamData:
            if x.is_cached:
                transform_strategy = TransformStrategy.CC
            elif x.is_stored:
                transform_strategy = TransformStrategy.SC
            else:
                raise ValueError(f"BeamData is not cached or stored, check your configuration")

        if transform_strategy == TransformStrategy.S and type(x) == BeamData:
            if x.is_cached:
                transform_strategy = TransformStrategy.CS
            elif x.is_stored:
                transform_strategy = TransformStrategy.SS
            else:
                raise ValueError(f"BeamData is not cached or stored, check your configuration")

        if (transform_strategy in [TransformStrategy.CC, TransformStrategy.CS] and
                type(x) == BeamData and not x.is_cached):
            logger.warning(f"Data is not cached but the transformation strategy is {transform_strategy}, "
                           f"caching data for transformer: {self.name} before the split to chunks.")
            x.cache()

        if (transform_strategy in [TransformStrategy.SC, TransformStrategy.SS] and
                type(x) == BeamData and not x.is_stored):
            logger.warning(f"Data is not stored but the transformation strategy is {transform_strategy}, "
                           f"storing data for transformer: {self.name} before the split to chunks.")
            x.store()

        store_chunk = transform_strategy in [TransformStrategy.CS, TransformStrategy.SS, TransformStrategy.S]

        if store_path is None and store_chunk:

            if isinstance(x, BeamData) and x.path is not None:
                store_path = x.path
                store_path = store_path.parent.joinpath(f"{store_path.name}_transformed_{self.name}")
                logger.info(f"Path is not specified for transformer: {self.name}, "
                            f"the chunk will be stored in a neighboring directory as the original data: {x.path}"
                            f"to: {store_path}.")
            else:
                logger.warning(f"Path is not specified for transformer: {self.name}, "
                               f"the chunk will not be stored.")
                store_chunk = False
        elif store_chunk:
            logger.info(f"Storing transformed chunks of data in: {store_path}")
            if is_chunk:
                store_path.mkdir(parents=True, exist_ok=True)

        queue = BeamParallel(n_workers=n_workers, func=None, method=mp_method, name=self.name,
                             progressbar='beam', reduce=False, reduce_dim=reduce_dim, use_dill=use_dill,
                             **parallel_kwargs)

        if return_results is None:
            return_results = not (store or store_chunk)

        sorted_keys = []
        if is_chunk:
            logger.info(f"Splitting data to chunks for transformer: {self.name}")
            for k, c in tqdm(self.chunks(x, chunksize=chunksize, n_chunks=n_chunks,
                                         squeeze=squeeze, split_by=split_by, partition=partition,
                                         chunksize_policy=chunksize_policy)):

                sorted_keys.append(k)
                chunk_path = None
                if store_chunk:

                    k_type = check_type(k)
                    k_with_counter = k
                    if k_type.element == Types.int:
                        k_with_counter = BeamData.normalize_key(k_with_counter)

                    chunk_path = store_path.joinpath(f"{self.name}_{k_with_counter}{part_name}")

                    if store_suffix is not None:
                        chunk_path = chunk_path.with_suffix(chunk_path.suffix + store_suffix)

                queue.add(BeamTask(self.worker, c, key=k, is_chunk=is_chunk, store_path=chunk_path,
                                   override=override, store=store_chunk, name=k, metadata=f"{self.name}",
                                   return_results=return_results, use_cache=use_cache, task_kwargs=kwargs,
                                   silent=silent, retries=retries, retries_delay=retries_delay))

        else:

            if store_path:

                store_path = store_path.joinpath(f"{self.name}_{self.counter}")
                if store_suffix is not None:
                    store_path = store_path.with_suffix(store_path.suffix + store_suffix)

            queue.add(BeamTask(self.worker, x, key=None, is_chunk=is_chunk, store_path=store_path,
                               override=override, store=store_chunk, name=self.name, return_results=return_results,
                               task_kwargs=kwargs))

        self.counter += 1
        logger.info(f"Starting transformer: {self.name} with {n_workers} workers. "
                    f"Number of queued tasks is {len(queue)}.")

        synced_results = queue.run(n_workers=n_workers, method=mp_method, shuffle=shuffle)

        exceptions = []
        for i, (_, v) in enumerate(iter_container(synced_results.exceptions)):
            exceptions.append({**v, 'task': queue.queue[i]})

        if len(exceptions) > 0:
            logger.error(f"Transformer {self.name} had {len(exceptions)} exceptions during operation.")
            logger.info("Failed tasks can be obtained in self.exceptions")
            self.exceptions = exceptions

        results = synced_results.results

        if is_chunk:
            values = [xi.result[1] if xi.exception is None else xi for xi in results]
            keys = [xi.name for xi in results]
            keys = [ki if type(ki) is tuple else (ki,) for ki in keys]
            sorted_keys = [ki if type(ki) is tuple else (ki,) for ki in sorted_keys]
            y = build_container_from_tupled_keys(keys, values, sorted_keys=sorted_keys)

            if len(exceptions) == 0:

                logger.info(f"Finished transformer process: {self.name}. Collating results...")

                if reduce and not (store_chunk and not store):
                    y = self.reduce(y, split_by=split_by, x_in=x)
            else:
                # x = {k[0] if type(k) is tuple and len(k) == 1 else k: v for k, v in zip(keys, values)}
                if store and not store_chunk:
                    logger.warning("Due to exceptions, the data will not be stored, "
                                   "the data is returned as a dictionary of all the successful tasks.")
                if strict:
                    logger.error(exceptions)
                    raise Exception("Exceptions occurred during the transformation, the strict mode is enabled.")

                return y

        else:
            if len(exceptions) > 0:
                logger.warning("Exception occurred, the exception object and the task are returned.")
                if strict:
                    logger.error(exceptions)
                    raise Exception("Exceptions occurred during the transformation, the strict mode is enabled.")

                return results
            logger.info(f"Finished transformer process: {self.name}.")
            y = results[0].result[1]

        if store and not store_chunk:

            logger.info(f"Storing aggregated transformed data in: {store_path}")
            if not isinstance(y, BeamData):
                y = BeamData(y)
            y.store(path=store_path)

        if return_results:
            return y
        elif store or store_chunk:
            return
        else:
            return y


