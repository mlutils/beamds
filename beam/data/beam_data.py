import time
from collections import defaultdict
from copy import deepcopy
from functools import partial, cached_property

import numpy as np
import pandas as pd

from ..logging import beam_logger as logger
from ..path import beam_path, prioritized_extensions

from ..base import Groups, Iloc, Loc, Key, return_none
from ..meta import BeamName
from ..type import BeamType, is_beam_processor, is_beam_data, Types
from ..utils import (is_container, Slicer, recursive, iter_container, recursive_collate_chunks,
                     collate_chunks, recursive_flatten, recursive_flatten_with_keys, recursive_device,
                     container_len, recursive_len, is_arange, recursive_size, divide_chunks,
                     recursive_hierarchical_keys,
                     recursive_types, recursive_shape, recursive_slice, recursive_slice_columns, recursive_batch,
                     get_closest_item_with_tuple_key, get_item_with_tuple_key, set_item_with_tuple_key,
                     recursive_chunks, as_numpy, check_type, as_tensor, slice_to_index, beam_device, beam_hash,
                     DataBatch, recursive_same_device, recursive_concatenate, recursive_keys, concat_polars_horizontally, beam_traceback)


class BeamData(BeamName):

    # metadata files
    metadata_files = {'conf': '.conf.pkl', 'schema': '.schema.pkl',
                      'info': '.info.fea', 'label': '.label', 'aux': '.aux.pkl',
                      'index': '.index', 'all_paths': '.all_paths.pkl'}

    default_data_file_name = 'data_container'
    index_chunk_file_extension = '.index_chunk'
    columns_chunk_file_extension = '.columns_chunk'
    index_partition_directory_name = '.index_part'
    columns_partition_directory_name = '.columns_part'

    def __init__(self, *args, data=None, path=None, name=None, all_paths=None,
                 index=None, label=None, columns=None, lazy=True, device=None, target_device=None, schema=None,
                 override=False, compress=None, split_by='keys', chunksize=int(1e9), chunklen=None, n_chunks=None,
                 key_map=None, partition=None, archive_size=int(1e6), preferred_orientation='index', read_kwargs=None,
                 write_kwargs=None, quick_getitem=False, orientation=None, glob_filter=None, info=None, synced=False,
                 write_metadata=True, read_metadata=True, metadata_path_prefix=None, key_fold_map=None,
                 chunksize_policy='round', **kwargs):

        '''

        @param args:
        @param data:
        @param path: if not str, requires to support the pathlib Path attributes and operations, can be container of paths
        @param lazy:
        @param kwargs:
        @param split_by: 'keys', 'index', 'columns'. The data is split to chunks by the keys, index or columns.

        Possible orientations are: row/column/other

        There are 4 possible ways of data orientation:

        1. simple: simple representation of tabular data where there is only a single data array in self.data.
        This orientation should support the fastest getitem operations

        2. columns: in this orientation, each data element represents different set of columns or information about the
        same data so each data element has the same length and each row in each data element corresponds to the same object.

        3. index:  in this orientation the rows are spread over different data elements so the data elements may have
         different length but their shape[1:] is identical, so we can collect batch of elements and concat them together.

         4. packed: each data element represents different set of data points but each data point may have different nature.
         this could model for example node properties in Knowledge graph where there are many types of nodes with different
         features. In case there is a common index, it can be used to slice and collect the data like the original
         PackedFold object.

         If data is both cached in `self.data` and stored in self.all_paths, the cached version is always preferred.

        The orientation is inferred from the data. If all objects have same length they are assumed to represent columns
        orientation. If all objects have same shape[1:] they are assumed to represent index orientation. If one wish to pass
        an index orientation data where all objects have same length, one can pass the preferred_orientation='index' argument.

        '''

        #todo: add support for target device+to_tensor when returning DataBatch
        super().__init__(name=name)

        if synced and path is None:
            raise ValueError("Synced mode requires a path")

        self.lazy = lazy if synced is False else False
        self.synced = synced
        self.override = override
        self.compress = compress
        self.chunksize = chunksize
        self.chunklen = chunklen
        self.n_chunks = n_chunks
        self.partition = partition
        self.archive_size = archive_size
        self.target_device = beam_device(target_device)
        self._index = index
        self._label = label
        self._schema = schema
        self.columns = columns
        self.preferred_orientation = preferred_orientation
        self.split_by = split_by
        self.quick_getitem = quick_getitem
        self.glob_filter = glob_filter
        self._key_map = key_map
        self.write_metadata = write_metadata
        self.read_metadata = read_metadata
        self.metadata_path_prefix = beam_path(metadata_path_prefix)
        self.key_fold_map = key_fold_map
        self.chunksize_policy = chunksize_policy

        self.is_stored = False
        self.is_cached = True

        self._columns_map = None
        self._device = None
        self._len = None
        self._data_types = None
        self._data_type = None
        self._objects_type = None
        self._schema_type = None
        self._flatten_data = None
        self._flatten_items = None
        self._size = None
        self._total_size = None
        self._conf = None
        self._info_groups = None
        self._all_paths = None
        self._root_path = None
        self._metadata_paths = None
        self._has_index = None
        self._has_label = None
        self._metadata_path_exists = {}
        self.groups = Groups(self.get_info_groups)

        self._info = info
        self._orientation = orientation

        self.iloc = Iloc(self)
        self.loc = Loc(self)
        self.key = Key(self)

        self.read_kwargs = {} if read_kwargs is None else read_kwargs
        self.write_kwargs = {} if write_kwargs is None else write_kwargs

        # first we check if the BeamData object is cached (i.e. the data is passed as an argument)
        if len(args) == 1:
            self.data = args[0]
        elif len(args):
            self.data = list(args)
        elif len(kwargs):
            self.data = kwargs
        elif data is not None:
            self.data = data
        else:
            self.data = None
            self.is_cached = False
            # in this case the data is not cached, so it should be stored ether in root_path or in all_paths

        if device is not None:
            device = beam_device(device)
            self.as_tensor(device=device)
            self._device = device

        path = beam_path(path)

        self._name = name

        if path is not None:
            self._name = path.name
        else:
            self._name = None

        if is_container(path):
            self._all_paths = path
        elif path is not None:

            self._root_path, self._all_paths, self._metadata_paths = self.single_file_case(path, all_paths,
                                                                                               self._metadata_paths)

        if ((self._all_paths is not None) or (self._root_path is not None and
                                              self._root_path.not_empty(filter_pattern=r'^\.'))) and \
                not self.is_cached:
            self.is_stored = True
            if not self.lazy:
                self.cache()

    @cached_property
    def data_slicer(self):
        return Slicer(self.data)

    @cached_property
    def index_slicer(self):
        return Slicer(self.index)

    @cached_property
    def label_slicer(self):
        return Slicer(self.label)

    @property
    def metadata_paths(self):

        if self._metadata_paths is not None:
            return self._metadata_paths

        root_path = self.root_path
        if self.metadata_path_prefix is not None:
            if self.metadata_path_prefix.is_absolute():
                root_path = self.metadata_path_prefix
            else:
                root_path = self.root_path.joinpath(self.metadata_path_prefix)

        self._metadata_paths = {k: root_path.joinpath(v) for k, v in BeamData.metadata_files.items()}
        return self._metadata_paths

    @property
    def root_path(self):
        if self._root_path is not None:
            return self._root_path

        if self.all_paths is not None:
            self._root_path = self.recursive_root_finder(self.all_paths)
            return self._root_path

        return self._root_path

    @staticmethod
    def clear_metadata(path):
        path = beam_path(path)
        if path.is_file():
            name = path.stem
            if name in ['.all_paths', '.info', '.conf']:
                path.unlink()
        elif path.is_dir():
            for new_path in list(path):
                BeamData.clear_metadata(new_path)

    @property
    def all_paths(self):

        if self._all_paths is not None:
            return self._all_paths

        if self.is_stored:
            path = self.metadata_paths['all_paths']
            if self.read_metadata and path.exists():
                logger.debug(f"Reading all_paths file: {path}")
                self._all_paths = path.read()

            else:
                self._all_paths = BeamData.recursive_map_path(self.root_path, glob_filter=self.glob_filter)
                if self.write_metadata:
                    BeamData.write_file(self._all_paths, path)

        return self._all_paths

    def get_info_groups(self):
        if self._info_groups is not None:
            return self._info_groups
        self._info_groupby = self.info.groupby('fold')
        return self._info_groupby

    @property
    def has_index(self):
        if self._has_index is None:
            _ = self.index
        self._has_index = self._index is not None and len(self._index)
        return self._has_index

    @property
    def has_label(self):
        if self._has_label is None:
            _ = self.label
        self._has_label = self._label is not None
        return self._has_label

    @property
    def index(self):
        if self._index is not None:
            return self._index

        if self.is_stored and self.metadata_paths['index'].parent.is_dir():

            for path in self.metadata_paths['index'].parent.iterdir():
                if path.stem == BeamData.metadata_files['index']:
                    if path.exists():
                        logger.debug(f"Reading index file: {path}")
                        self._index = path.read()
                        return self._index
        if self.is_cached:
            info = self.info
            if self.orientation is None:
                self.clear_index()
            elif self.orientation in ['columns', 'simple']:
                self._index = info.index.values
            elif self.orientation == 'index':
                @recursive
                def replace_key_map_with_index(ind):
                    return self.info[self.info.fold == ind].index.values
                self._index = replace_key_map_with_index(deepcopy(self.key_map))
            elif self.orientation == 'packed':
                # no consistent definition of index for packed case
                self.clear_index()
            else:
                raise ValueError(f"Unknown orientation: {self.orientation}")

        if self._index is not None and self.objects_type == Types.tensor:
            self._index = as_tensor(self._index, device=self.device or 'cpu')

        return self._index

    @property
    def label(self):

        if self._label is not None:
            return self._label

        if self.is_stored and self.metadata_paths['label'].parent.is_dir():

            for path in self.metadata_paths['label'].parent.iterdir():
                if path.stem == BeamData.metadata_files['label']:
                    if path.exists():
                        logger.debug(f"Reading label file: {path}")
                        self._label = path.read()
                        return self._label

        if self._label is not None and self.objects_type == Types.tensor:
            self._label = as_tensor(self._label, device=self.device or 'cpu')

        return self._label

    def single_file_case(self, root_path, all_paths, metadata_paths):

        single_file = False
        if all_paths is None and not root_path.is_root() and root_path.parent.is_dir():

            if root_path.is_file():
                single_file = True

            for p in root_path.parent.iterdir():
                if p.stem == root_path.stem and p.is_file():
                    single_file = True
                    root_path = p
                    break

        if single_file:
            meta_root_path = root_path.parent
            if self.metadata_path_prefix is not None:
                if self.metadata_path_prefix.is_absolute():
                    meta_root_path = self.metadata_path_prefix
                else:
                    meta_root_path = meta_root_path.joinpath(self.metadata_path_prefix)

            meta_path = meta_root_path.joinpath(f'.{root_path.name}')
            metadata_paths = {k: meta_path.joinpath(v) for k, v in BeamData.metadata_files.items()}
            all_paths = {BeamData.default_data_file_name: root_path.name}
            root_path = root_path.parent

        return root_path, all_paths, metadata_paths

    @staticmethod
    def collate(*args, batch=None, split_by=None, **kwargs):

        if len(args) == 1:
            batch = args[0]
        elif len(args):
            batch = list(args)
        elif len(kwargs):
            batch = kwargs

        k, bd = next(iter_container(batch))

        orientation = bd.orientation
        if orientation == 'index':
            columns = bd.columns
        else:
            columns = None

        if split_by is None:
            split_by = bd.split_by

        if split_by == 'columns':
            dim = 1
            squeeze = True
        elif split_by == 'index':
            dim = 0
            squeeze = True
        else:
            dim = None
            squeeze = False

        @recursive
        def get_data(x):
            if isinstance(x, BeamData):
                return x.data
            if type(x) is list:
                x = [xi.data for xi in x]
                if squeeze:
                    x = recursive_collate_chunks(*x, dim=dim)
            return x

        @recursive
        def get_index(x):
            if isinstance(x, BeamData):
                return x._index
            if type(x) is list:
                x = [xi.index for xi in x]
                if squeeze and dim == 0:
                    x = collate_chunks(*x, dim=0)
            return x

        @recursive
        def get_label(x):
            if isinstance(x, BeamData):
                return x._label
            if type(x) is list:
                x = [xi.label for xi in x]
                if squeeze and dim == 0:
                    x = collate_chunks(*x, dim=0)
            return x

        data = get_data(batch)
        index = get_index(batch)
        label = get_label(batch)

        return bd.clone(data, columns=columns, index=index, label=label, key_fold_map=None)

    @classmethod
    def from_path(cls, path, *args, **kwargs):
        path = beam_path(path)
        if not path.exists():
            path.mkdir()
        return cls(path=path, *args, **kwargs)

    def to_path(self, path):
        self.store(path=path)

    @classmethod
    def from_indexed_pandas(cls, data, *args, **kwargs):

        @recursive
        def get_index(x):
            return x.index

        index = get_index(data)
        kwargs['index'] = index

        return cls(data, *args, **kwargs)

    @property
    def objects_type(self):

        if self._objects_type is not None:
            return self._objects_type

        objects_types = recursive_flatten(self.data_types)
        objects_types = [v.minor for v in objects_types if v.minor != Types.none]

        u = set(objects_types)

        if len(u) == 1:
            self._objects_type = next(iter(u))
        else:
            self._objects_type = 'mixed'

        return self._objects_type

    @property
    def schema(self):

        if self._schema is not None:
            return self._schema

        if self.is_stored:
            if self.metadata_path_exists('schema'):
                schema_path = self.metadata_paths['schema']
                logger.debug(f"Reading schema file {schema_path}")
                self._schema = schema_path.read()
                return self._schema

        return self._schema

    @property
    def conf(self):

        if self._conf is not None:
            return self._conf

        if self.is_stored and self.read_metadata:
            if self.metadata_path_exists('conf'):
                conf_path = self.metadata_paths['conf']
                logger.debug(f"Reading conf file {conf_path}")
                self._conf = conf_path.read()
                return self._conf

        if self.is_cached:
            self._conf = {'orientation': self.orientation,
                          'objects_type': self.objects_type,
                          'len': len(self),
                          'columns': self.columns,
                          'device': self.device,
                          'has_index': self._index is not None,
                          'has_label': self._label is not None,
                          'time': time.time(),
                          'columns_map': self.columns_map}
            return self._conf

        self._conf = None
        return defaultdict(return_none)

    @property
    def key_map(self):
        if self._key_map is not None:
            return self._key_map
        self._key_map = {k: i for i, (k, v) in enumerate(self.flatten_items.items())}

        return self._key_map

    def metadata_path_exists(self, key):
        if key not in self._metadata_path_exists:
            self._metadata_path_exists[key] = self.metadata_paths[key].is_file()
        return self._metadata_path_exists[key]

    @property
    def info(self):

        if self._info is not None:
            return self._info

        if self.is_stored and self.read_metadata:
            if self.metadata_path_exists('info'):
                info_path = self.metadata_paths['info']
                logger.debug(f"Reading info file {info_path}")
                self._info = info_path.read()
                return self._info

        if self.is_cached:

            if self.orientation in ['index', 'packed']:
                filtered_data = list(filter(lambda x: x is not None, self.flatten_data))
                if not len(filtered_data):
                    self._info = None
                    return self._info

                # fold_index = np.concatenate([np.arange(len(d)) if hasattr(d, '__len__')
                #                              else np.array([0]) for d in filtered_data])
                # fold = np.concatenate([np.full(len(d), k) if hasattr(d, '__len__')
                #                        else np.array([k]) for k, d in enumerate(filtered_data)])

                fold_index = []
                fold = []
                lengths = []
                for k, d in enumerate(filtered_data):
                    if hasattr(d, '__len__') and BeamType.check(d, major=False, minor=True, element=False).is_data_array:
                        fold_index.append(np.arange(len(d)))
                        fold.append(np.full(len(d), k))
                        lengths.append(len(d))
                    else:
                        fold_index.append(np.array([0]))
                        fold.append(np.array([k]))
                        lengths.append(1)

                fold_index = np.concatenate(fold_index)
                fold = np.concatenate(fold)
                lengths = np.array(lengths)
                # still not sure if we really need this column. if so, it should be fixed
                # fold_key = np.concatenate([np.full(len(d), k) for k, d in self.flatten_items.items()])

                # lengths = np.array([len(d) if hasattr(d, '__len__') else 1
                #                     for d in self.flatten_data])

                offset = np.cumsum(lengths, axis=0) - lengths
                offset = offset[fold] + fold_index

            else:
                fold_index = None
                fold = None
                offset = None
                # fold_key = None

            if self._index is not None:
                # it is assumed that if orientation is in ['columns', 'simple'], then _index is a single array
                index = np.concatenate([as_numpy(i) for i in recursive_flatten([self._index])])
            else:
                # assert len(self) == len(offset)
                index = np.arange(len(self))

            if self._label is not None:
                label = np.concatenate([as_numpy(l) for l in recursive_flatten([self._label])])
            else:
                label = None

            info = {'fold': fold, 'fold_index': fold_index,
                    # 'fold_key': fold_key,
                    'offset': offset,
                    'map': np.arange(len(index))}

            if self._label is not None:
                info['label'] = label

            self._info = pd.DataFrame(info, index=index)

            # try:
            #     self._info = pd.DataFrame(info, index=index)
            # except Exception as e:
            #     if self.orientation == 'packed':
            #         logger.warning(f"Error creating info DataFrame: {e}, returning simplified version")
            #         self._len = len(self.data)
            #         self._info = pd.DataFrame({'fold': np.zeros(self._len, dtype=int),
            #                                    'fold_index': np.arange(self._len),
            #                                    'offset': np.arange(self._len),
            #                                    'map': np.arange(self._len)},
            #                                   index=np.arange(self._len))
            #     else:
            #         raise e

            return self._info

        self._info = None
        return self._info

    @property
    def path(self):
        if self.all_paths is not None and BeamData.default_data_file_name in self.all_paths and len(self.all_paths) == 1:
            return self.root_path.joinpath(self.all_paths[BeamData.default_data_file_name])
        return self.root_path

    @schema.setter
    def schema(self, schema):
        self._schema = schema

    @path.setter
    def path(self, value):
        if self.root_path is not None:
            logger.warning(f'path already set to {self.root_path}, overwriting with {value}')
        value = beam_path(value)
        if value.is_dir() and len(list(value.iterdir())):
            raise ValueError(f'path {value} is not empty')
        self._root_path = value
        self._all_paths = None
        self._metadata_paths = None
        self.is_stored = False


    @cached_property
    def index_type(self):
        return check_type(self.index)

    @cached_property
    def label_type(self):
        return check_type(self.label)

    @staticmethod
    def normalize_key(key):
        if type(key) is tuple:
            key = '/'.join([BeamData.normalize_key(k) for k in key])
        if type(key) is not str:
            key = f'{key:06}'
        return key

    @property
    def flatten_data(self):
        if self._flatten_data is not None:
            return self._flatten_data
        self._flatten_data = recursive_flatten(self.data)
        return self._flatten_data

    @property
    def flatten_items(self):
        if self._flatten_items is not None:
            return self._flatten_items
        flatten_items = recursive_flatten_with_keys(self.data)
        for k in list(flatten_items.keys()):
            if len(k) == 1:
                flatten_items[k[0]] = flatten_items.pop(k)

        self._flatten_items = flatten_items
        return self._flatten_items

    @property
    def device(self):
        if self._device is not None:
            return self._device

        if self.objects_type == Types.tensor:
            # "All tensors should be on the same device"
            if recursive_same_device(self.data):
                self._device = recursive_device(self.data)

        return self._device

    def to(self, device):
        self.data = recursive(lambda x: x.to(device))(self.data)
        self._device = device
        return self

    def __len__(self):

        if self._len is not None:
            return self._len

        if self._info is not None:
            self._len = len(self._info)
            return self._len

        if self.is_stored and self._conf is not None:
            self._len = self._conf['len']
            return self._len

        if self.is_cached:
            if self.orientation == 'columns':
                _len = container_len(self.data)
            else:

                _len = recursive_len(self.data, data_array_only=True)
                _len = sum(recursive_flatten([_len], flat_array=True))

            self._len = _len
            return self._len

        self._len = None
        return self._len

    @property
    def orientation(self):

        if self._orientation is not None:
            return self._orientation

        if self._conf is not None:
            self._orientation = self._conf['orientation']
            return self._orientation

        if self.is_cached:

            if self.data is None:
                return None

            if not is_container(self.data):
                self._orientation = 'simple'
                if self.data_type.minor in [Types.pandas, Types.polars, Types.cudf] and self.columns is None:
                    self.columns = self.data.columns
                if self.data_type.minor in [Types.pandas, Types.cudf] and self._index is None:
                    self._index = self.data.index

            else:

                def shape_of(x):
                    if not BeamType.check_if_data_array(x):
                        return 'other'
                    if hasattr(x, 'shape'):
                        return tuple(x.shape[1:])
                    if x is None:
                        return None
                    if hasattr(x, '__len__'):
                        return ()
                    return 'scalar'

                lens = recursive_flatten(recursive_len([self.data]), flat_array=True)
                lens = set(list(filter(lambda x: x != 0, lens)))

                lens_index = recursive_flatten(recursive_len([self._index]), flat_array=True)
                lens_index = set(list(filter(lambda x: x is not None, lens_index)))

                if len(lens) == 1 and len(lens_index) <= 1:
                    if self.preferred_orientation == 'columns':
                        self._orientation = 'columns'
                    else:
                        shapes = recursive_flatten(recursive(shape_of)([self.data]))
                        shapes = set(list(filter(lambda x: x is not None, shapes)))
                        if len(shapes) > 1 and 'other' not in shapes:
                            self._orientation = 'columns'
                        elif len(shapes) == 1 and 'other' not in shapes:
                            self._orientation = 'index'
                        else:
                            self._orientation = 'packed'
                else:
                    shapes = recursive_flatten(recursive(shape_of)([self.data]))
                    shapes = list(filter(lambda x: x is not None, shapes))

                    if len(set(shapes)) == 1 and shapes[0] != 'other' and len(shapes[0]):
                        self._orientation = 'index'
                    else:
                        self._orientation = 'packed'

        elif self.is_stored:
            self._orientation = self.conf['orientation']

        else:
            if self.data is None:
                return None
            self._orientation = 'packed'

        return self._orientation

    def set_property(self, p):
        setattr(self, f"_{p}", None)
        return getattr(self, p)

    @property
    def data_types(self):
        if self._data_types is not None:
            return self._data_types
        self._data_types = recursive(check_type)(self.data)
        return self._data_types

    @property
    def data_type(self):
        if self._data_type is not None:
            return self._data_type
        self._data_type = check_type(self.data)
        return self._data_type

    @staticmethod
    def write_file(data, path, override=True, schema=None, **kwargs):

        if schema is not None:
            kwargs = {**schema.write, **kwargs}

        path = beam_path(path)

        if (not override) and path.exists():
            raise NameError(f"File {path} exists. "
                            f"Please specify write_file(...,overwrite=True) to write on existing file")

        path.clean()
        logger.debug(f"Writing file: {path}")
        path = path.write(data, **kwargs)

        return path

    @staticmethod
    def get_schema_from_subset(schema, key, schema_type=None):

        if schema_type is None:
            schema_type = check_type(schema)

        if schema_type.minor == Types.dict and key in schema:
            s = schema[key]
        elif schema_type.minor == Types.list and key < len(schema):
            s = schema[key]
        elif schema_type.major == Types.container:
            s = None
        else:
            s = schema
        return s

    @staticmethod
    def get_schema_from_tupled_key(schema, key, schema_type=None):
        for k in key:
            schema = BeamData.get_schema_from_subset(schema, k, schema_type=schema_type)
        return schema

    @staticmethod
    def containerize_keys_and_values(keys, values):

        argsort, isarange = is_arange(keys)
        if not isarange:
            values = dict(zip(keys, values))
            # values = {k: values[k] for k in sorted(values.keys())}
        else:
            values = [values[i] for i in argsort]

        if type(values) is dict and BeamData.default_data_file_name in values:
            if len(values) == 1:
                values = values[BeamData.default_data_file_name]
            else:
                d = values.pop(BeamData.default_data_file_name)
                if type(d) is dict:
                    values = {**values, **d}
                else:
                    values[BeamData.default_data_file_name] = d

        return values

    @staticmethod
    def recursive_root_finder(all_paths, head=None):
        if head is None:
            head = []

        if is_container(all_paths):

            k, v = next(iter_container(all_paths))
            head.append(k)
            return BeamData.recursive_root_finder(v, head=head)

        if all_paths.is_file():
            return all_paths.parent.joinpath(all_paths.stem)

        for _ in head:
            all_paths = all_paths.parent

        return all_paths

    @staticmethod
    def recursive_map_path(root_path, relative_path=None, glob_filter=None):

        if relative_path is None:
            relative_path = ''
            path = root_path
        else:
            path = root_path.joinpath(relative_path)

        if path.is_dir():

            keys = []
            keys_paths = []
            values = []

            if glob_filter is not None:

                if hasattr(path, 'glob'):
                    path_list = path.glob(glob_filter)
                else:
                    logger.warning(f"Path {path} does not support glob method. Skipping glob filter {glob_filter}.")
                    path_list = path.iterdir()
            else:
                path_list = path.iterdir()

            for next_path in path_list:

                # skip hidden files which we use for metadata (see BeamData.metadata_files)
                if next_path.name.startswith('.'):
                    continue

                k = next_path.stem if next_path.is_file() else next_path.name
                keys.append(k)
                keys_paths.append(next_path)
                rp = str(next_path.relative_to(root_path))
                values.append(BeamData.recursive_map_path(root_path, relative_path=rp, glob_filter=glob_filter))

            if len(keys) == 0:
                return None

            # if the directory contains chunks it is considered as a single path
            if all([BeamData.index_chunk_file_extension in p.name for p in keys_paths]):
                return relative_path
            elif all([BeamData.columns_chunk_file_extension in p.name for p in keys_paths]):
                return relative_path

            argsort, isarange = is_arange(keys)

            if not isarange:
                values = dict(zip(keys, values))
            else:
                values = [values[i] for i in argsort]

            if type(values) is dict and BeamData.default_data_file_name in values and len(values) == 1:
                values = values[BeamData.default_data_file_name]

            return values

        # we store the files without their extension? why?
        # if path.is_file():
        #     return path.parent.joinpath(path.stem)
        if path.is_file():
            return relative_path

        return None

    def as_tensor(self, device=None, dtype=None, return_vector=False):

        '''
        Convert the data to tensor in place
        @param device:
        @param dtype:
        @param return_vector:
        @return:
        '''

        func = partial(as_tensor, device=device, dtype=dtype, return_vector=return_vector)
        self.data = recursive(func)(self.data)
        self._index = func(self._index)
        self._label = func(self._label)
        self._objects_type = Types.tensor

        return self

    def as_numpy(self):

        '''
        Convert the data to numpy in place
        @return:
        '''

        func = partial(as_numpy)
        self.data = recursive(func)(self.data)
        self._index = func(self._index)
        self._label = func(self._label)
        self._objects_type = Types.numpy

        return self

    @property
    def values(self):

        if not self.is_cached:
            bd = self.cache(in_place=True)
        else:
            bd = self

        return bd.data

    @staticmethod
    def exists(paths):
        return BeamData.read(paths, _check_existence=True)

    @staticmethod
    def read(paths, schema=None, strict=False, _check_existence=False, **kwargs):

        if paths is None:
            if strict:
                raise ValueError("No path provided")
            return None

        if is_container(paths):
            keys = []
            values = []

            schema_type = check_type(schema)

            for k, next_path in iter_container(paths):

                s = BeamData.get_schema_from_subset(schema, k, schema_type=schema_type)
                values.append(BeamData.read(next_path, schema=s, _check_existence=_check_existence, **kwargs))
                keys.append(k)

            return BeamData.containerize_keys_and_values(keys, values)

        path = beam_path(paths)
        if schema is not None:
            kwargs = {**schema.read_schema, **kwargs}

        if path.is_file() or path.suffix in ['.bmpr', '.bmd']:
            if _check_existence:
                return True
            logger.debug(f"Reading file: {path}")
            return path.read(**kwargs)

        if path.is_dir():

            if _check_existence:
                return True

            keys = []
            values = []

            for next_path in path.iterdir():

                if not next_path.name.startswith('.'):
                    keys.append(next_path.stem)
                    values.append(BeamData.read(next_path, schema=schema, **kwargs))

            if all([BeamData.index_chunk_file_extension in k for k in keys]):
                return collate_chunks(*values, keys=keys, dim=0)
            elif all([BeamData.columns_chunk_file_extension in k for k in keys]):
                return collate_chunks(*values, keys=keys, dim=1)
            elif all([BeamData.index_partition_directory_name in k for k in keys]):
                return recursive_collate_chunks(*values, dim=0)
            elif all([BeamData.columns_partition_directory_name in k for k in keys]):
                return recursive_collate_chunks(*values, dim=1)
            else:
                return BeamData.containerize_keys_and_values(keys, values)

        if path.parent.is_dir():
            for i, p in enumerate(path.parent.iterdir()):
                if p.stem == path.stem:
                    if _check_existence:
                        return True
                    return p.read(**kwargs)
                if len(prioritized_extensions) > i:
                    path_with_suffix = path.with_suffix(prioritized_extensions[i])
                    if path_with_suffix.exists():
                        if _check_existence:
                            return True
                        return path_with_suffix.read(**kwargs)

        if _check_existence:
            return False
        elif strict:
            raise ValueError(f"No object found in path: {path}")
        else:
            logger.warning(f"No object found in path: {path}")
            return None

    @staticmethod
    def write_tree(data, path, sizes=None, split_by='keys', archive_size=int(1e6), chunksize=int(1e9), override=True,
                   chunklen=None, n_chunks=None, partition=None, file_type=None, root=False, schema=None,
                   split=False, textual_serialization=False, blacklist_priority=None, chunksize_policy='round', **kwargs):

        path = beam_path(path)

        if sizes is None:
            sizes = recursive_size(data)

        if is_container(data):

            size_summary = sum(recursive_flatten(sizes, flat_array=True))

            if size_summary < archive_size:

                if root:
                    path = path.joinpath(BeamData.default_data_file_name)

                BeamData.write_object(data, path, size=size_summary, archive=True,
                                      blacklist_priority=blacklist_priority, **kwargs)
                return

            schema_type = check_type(schema)
            for k, v in iter_container(data):

                s = BeamData.get_schema_from_subset(schema, k, schema_type=schema_type)
                BeamData.write_tree(v, path.joinpath(BeamData.normalize_key(k)), sizes=sizes[k],
                                    archive_size=archive_size, chunksize=chunksize, chunklen=chunklen,
                                    split_by=split_by, n_chunks=n_chunks, partition=partition, root=False,
                                    file_type=file_type, schema=s, override=override, split=split,
                                    textual_serialization=textual_serialization,
                                    blacklist_priority=blacklist_priority, chunksize_policy=chunksize_policy, **kwargs)

        else:

            if root:
                path = path.joinpath(BeamData.default_data_file_name)

            BeamData.write_object(data, path, size=sizes, archive=False, override=override,
                                  chunksize=chunksize, chunklen=chunklen, split_by=split_by,
                                  n_chunks=n_chunks, partition=partition, schema=schema,
                                  file_type=file_type, split=split, textual_serialization=textual_serialization,
                                  blacklist_priority=blacklist_priority, chunksize_policy=chunksize_policy, **kwargs)

    @staticmethod
    def write_object(data, path, override=True, size=None, archive=False, compress=None, chunksize=int(1e9),
                     chunklen=None, n_chunks=None, partition=None, file_type=None, schema=None,
                     textual_serialization=False, split_by=None, split=True, priority=None,
                     blacklist_priority=None, chunksize_policy='round', **kwargs):

        path = beam_path(path)

        if not override:
            if path.exists() or (path.parent.is_dir() and any(p.stem == path.stem for p in path.parent.iterdir())):
                logger.warning(f"path {path} exists. To override, specify override=True")
                return
        else:
            path.clean()

        if archive:
            object_path = BeamData.write_file(data, path.with_suffix('.pkl'), override=override,
                                              schema=schema, **kwargs)
        else:

            if split and split_by != 'keys':
                n_chunks = BeamData.get_n_chunks(data, chunksize=chunksize, chunklen=chunklen,
                                                 n_chunks=n_chunks, size=size, chunksize_policy=chunksize_policy)
            else:
                n_chunks = 1

            data_type = check_type(data)
            if priority is None:
                priority = []
                if textual_serialization:
                    priority = ['.json', '.yaml']
                if partition is not None and data_type.minor == Types.pandas:
                    priority = ['.parquet', '.fea', '.pkl']
                elif partition is not None and data_type.minor == Types.polars:
                    priority = ['.pl.parquet', '.pl.fea', '.pl.pkl']
                elif data_type.minor == Types.pandas:
                    priority = ['.fea', '.parquet', '.pkl']
                elif data_type.minor == Types.polars:
                    priority.extend(['.pl.fea', '.pl.parquet', '.pl.pkl'])
                elif data_type.minor == Types.cudf:
                    priority = ['.cf.fea', '.cf.parquet', '.cf.pkl']
                elif data_type.minor == Types.numpy:
                    priority = ['.npy', '.pkl']
                elif data_type.minor == Types.scipy_sparse:
                    priority = ['.scipy_npz', '.pkl']
                elif data_type.minor == Types.tensor:
                    if data.is_sparse_csr:
                        priority = ['.pkl']
                    else:
                        priority = ['.pt']
                elif is_beam_data(data):
                    priority = ['.bmd', '.pkl', '.dill']
                elif is_beam_processor(data):
                    priority = ['.bmpr', '.pkl', '.dill', '.cloudpickle', '.joblib']
                else:
                    priority.extend(['.pkl', '.dill', '.cloudpickle', '.joblib'])

            if file_type is not None:
                priority.insert(file_type, 0)

            if blacklist_priority is not None:
                priority = [p for p in priority if p not in blacklist_priority]

            if split_by != 'keys' and n_chunks > 1:
                dim = {'index': 0, 'columns': 1}[split_by]
                data = list(divide_chunks(data, n_chunks=n_chunks, dim=dim))
                chunk_file_extension = {'index': BeamData.index_chunk_file_extension,
                                        'columns': BeamData.columns_chunk_file_extension}[split_by]
            else:
                data = [(0, data), ]
                chunk_file_extension = ''

            object_path = [None] * len(data)
            for i, di in data:

                if len(data) > 1:
                    path_i = path.joinpath(f"{i:06}{chunk_file_extension}")
                else:
                    path_i = path

                for ext in priority:
                    file_path = path_i.with_suffix(ext)
                    object_path[i] = file_path.name

                    try:
                        kwargs = {}
                        if ext == '.parquet':
                            if compress is False:
                                kwargs['compression'] = None
                            BeamData.write_file(di, file_path, partition_cols=partition, coerce_timestamps='us',
                                                allow_truncated_timestamps=True, schema=schema, **kwargs)
                        elif ext == '.fea':
                            if compress is False:
                                kwargs['compression'] = 'uncompressed'
                            BeamData.write_file(di, file_path, schema=schema, **kwargs)

                        elif ext == '.pkl':
                            if compress is False:
                                kwargs['compression'] = 'none'
                            BeamData.write_file(di, file_path, schema=schema, **kwargs)

                        elif ext == '.scipy_npz':
                            if compress is True:
                                kwargs['compressed'] = True
                            BeamData.write_file(di, file_path, schema=schema, **kwargs)

                        elif ext == '.bmpr':
                            BeamData.write_file(di, file_path, schema=schema, blacklist_priority=blacklist_priority,
                                                **kwargs)

                        else:
                            BeamData.write_file(di, file_path, schema=schema, **kwargs)

                        error = False
                        priority = [ext]
                        break

                    except Exception as e:
                        logger.warning(f"Failed to write file: {file_path.name}. Trying with the next file extension")
                        logger.debug(e)
                        logger.debug(beam_traceback())
                        error = True
                        if file_path.exists():
                            file_path.clean()

                if error:
                    logger.error(f"Could not write file: {path_i.name}.")

            if len(data) == 1:
                object_path = object_path[0]

        return object_path

    @property
    def parent(self):
        return BeamData.from_path(self.root_path.parent)

    @property
    def columns_map(self):

        if self._columns_map is not None:
            return self._columns_map

        if self.columns is not None:
            self._columns_map = {str(k): i for i, k in enumerate(self.columns)}

        self._columns_map = None
        return self._columns_map

    def keys(self, level=1):
        if self.is_cached:
            if type(self.data) is dict:
                for k in recursive_keys(self.data, level=level):
                    yield k
            else:
                for k in range(len(self.data)):
                    yield k
        else:
            if type(self.all_paths) is dict:
                for k in recursive_keys(self.all_paths, level=level):
                    yield k
            else:
                for k in range(len(self.all_paths)):
                    yield k

    def hierarchical_keys(self, recursive=False):
        if self.orientation is None:
            keys = []
        elif self.orientation == 'simple':
            keys = self.columns
        else:
            if self.is_cached:
                if recursive:
                    keys = recursive_hierarchical_keys(self.data)
                else:
                    if isinstance(self.data, dict):
                        keys = self.data.keys()
                    elif self.data is None:
                        return []
                    else:
                        keys = range(len(self.data))
            else:
                if recursive:
                    keys = recursive_hierarchical_keys(self.all_paths)
                else:
                    if isinstance(self.all_paths, dict):
                        keys = self.all_paths.keys()
                    else:
                        keys = range(len(self.all_paths))
        return keys

    def items(self, level=1):
        for k in self.keys(level=level):
            yield k, self[k]

    @property
    def dtypes(self):
        return recursive_types(self.data)

    @property
    def shape(self):
        return recursive_shape(self.data)

    @property
    def size(self):

        if self._size is not None:
            return self._size

        self._size = recursive_size(self.data)
        return self._size

    @staticmethod
    def _concatenate_values(data, orientation=None, objects_type=None):

        data = recursive_flatten(data)
        _, v = next(iter_container(data))

        if orientation is None:
            orientation = 'index'
        if objects_type is None:
            objects_type = check_type(v).minor

        if orientation == 'columns':
            dim = 1
        elif orientation == 'index':
            dim = 0
        else:
            return data

        if objects_type == Types.tensor:
            import torch
            func = torch.stack if dim == 1 and dim >= len(v.shape) else torch.cat
            kwargs = {'dim': dim}
        elif objects_type == Types.pandas:
            data = [pd.Series(v.values) if isinstance(v, pd.Index) else v for v in data]
            func = pd.concat
            kwargs = {'axis': dim}
        elif objects_type == Types.polars:
            if dim == 0:
                import polars as pl
                func = pl.concat
                kwargs = {'axis': dim}
            else:
                func = concat_polars_horizontally
                kwargs = {}
        elif objects_type == Types.cudf:
            import cudf
            func = cudf.concat
            data = [cudf.Series(v.values) if isinstance(v, cudf.Index) else v for v in data]
            kwargs = {'axis': dim}
        elif objects_type == Types.numpy:
            func = np.stack if dim==1 and dim >= len(v.shape) else np.concatenate
            kwargs = {'axis': dim}
        else:
            logger.warning(f"Concatenation not implemented for {objects_type}, returning the original data")
            return data

        return func(data, **kwargs)

    def concatenate_values(self, data=None, orientation=None, objects_type=None):

        if data is None:
            data = self.flatten_data
            orientation = self.orientation
            objects_type = self.objects_type

        return BeamData._concatenate_values(data, orientation=orientation, objects_type=objects_type)

    @staticmethod
    def concat(bds, dim=0):

        if len(bds) == 1:
            return bds[0]

        data = recursive_concatenate([d.data for d in bds], dim=dim)
        index = recursive_concatenate([d.index for d in bds], dim=0)
        label = recursive_concatenate([d.label for d in bds], dim=0)

        return bds[0].clone(data, index=index, label=label)

    def get_default_params(self, *args, **kwargs):
        """
        Get default parameters from the class

        @param args:
        @param kwargs:
        @return:
        """
        for k, v in kwargs.items():
            if hasattr(self, k) and v is None:
                kwargs[k] = getattr(self, k)

        for k in args:
            if hasattr(self, k):
                kwargs[k] = getattr(self, k)
            else:
                kwargs[k] = None

        return kwargs

    @property
    def total_size(self):

        if self._total_size is not None:
            return self._total_size
        self._total_size = sum(recursive_flatten(self.size, flat_array=True))
        return self._total_size

    def store(self, path=None, data=None, compress=None, chunksize=None,
              chunklen=None, n_chunks=None, partition=None, split_by=None,
              archive_size=None, override=None, split=True, chunksize_policy=None, **kwargs):

        override = override or self.override
        path = beam_path(path)

        if path is not None:
            if override:
                path.clean()
            self.path = path
        else:
            path = self.path

        sizes = None
        if data is None:
            data = self.data
            sizes = self.size

        path.clean()

        kwargs = self.get_default_params(compress=compress, chunksize=chunksize, chunklen=chunklen, n_chunks=n_chunks,
                                         partition=partition, split_by=split_by, archive_size=archive_size,
                                         chunksize_policy=chunksize_policy, **kwargs)

        BeamData.write_tree(data, path, root=True, sizes=sizes, schema=self.schema, override=override,
                            split=split, **kwargs)

        # store info and conf files
        if self.write_metadata:
            info_path = self.metadata_paths['info']
            BeamData.write_object(self.info, info_path)
            conf_path = self.metadata_paths['conf']
            BeamData.write_object({**self.conf}, conf_path, archive=True)
            conf_path = self.metadata_paths['conf']
            BeamData.write_object({**self.conf}, conf_path, archive=True)

        # store index and label
        if self.has_index:
            index_path = self.metadata_paths['index']
            BeamData.write_object(self.index, index_path)
        if self.has_label:
            label_path = self.metadata_paths['label']
            BeamData.write_object(self.label, label_path)

        self.is_stored = True
        self.data = data

        self._all_paths = BeamData.recursive_map_path(self.root_path, glob_filter=self.glob_filter)
        self.update_all_paths_file()

    def state_dict(self):
        if not self.is_cached:
            self.cache()

        return {'data': self.data, 'info': self.info, 'conf': self.conf, 'index': self.index, 'label': self.label,
                'schema': self.schema}

    def abs_all_paths(self, all_paths=None, root_path=None):

        if root_path is None:
            root_path = self.root_path

        @recursive
        def _abs_all_paths(path):
            if path is None:
                return None
            if isinstance(path, list):
                return [root_path.joinpath(p) for p in path]
            return root_path.joinpath(path)

        if all_paths is None:
            all_paths = self.all_paths

        return _abs_all_paths(all_paths)

    @classmethod
    def load_state_dict(cls, state_dict):
        return cls(**state_dict)

    def cached(self, *args, **kwargs):
        self.cache(*args, **kwargs)
        return self

    def cache(self, path=None, all_paths=None, schema=None, update=False, in_place=True, **kwargs):

        if self.is_cached:
            if update:
                logger.info(f"BeamData: Updating the cached data in path {self.path}")
            else:
                logger.info(f"BeamData: Data in path {self.path} is already cached. To update the cache use update=True")
                return

        if schema is None:
            schema = self.schema

        if all_paths is None and path is None:
            all_paths = self.all_paths

        if path is None:
            path = self.root_path

        if all_paths is None:
            all_paths = BeamData.recursive_map_path(path, glob_filter=self.glob_filter)

        # read the conf and info files

        if not self.is_stored:
            logger.warning("stored=False, data is seems to be un-synchronized")

        data = BeamData.read(self.abs_all_paths(all_paths), schema=schema, **kwargs)

        if in_place:
            self._root_path = path
            # self.all_paths = BeamData.recursive_map_path(root_path, glob_filter=self.glob_filter)
            self._all_paths = all_paths
            self.data = data
            self.is_stored = True
            self.is_cached = True
            self.reset_metadata()
        else:
            return BeamData(data=data, index=self.index, label=self.label)

        return self

    def reset_metadata(self, *args, avoid_reset=None):

        if avoid_reset is None:
            avoid_reset = []

        reset_params = ['_columns_map', '_device', '_len', '_data_types', '_data_type', '_objects_type',
                        '_info_groupby','_flatten_data', '_flatten_items', '_conf', '_info', '_orientation', '_size',
                        '_total_size', ]

        for param in reset_params:
            if param not in avoid_reset:
                setattr(self, param, None)

        for param in args:
            setattr(self, param, None)

        if 'metadata_path_exists' not in avoid_reset:
            self._metadata_path_exists = {}

    def inverse_map(self, ind):

        ind = slice_to_index(ind, sliced=self.index)

        index_type = check_type(ind)
        if index_type.major == Types.scalar:
            ind = [ind]

        return ind

    def __eq__(self, other):
        return self.values.__eq__(other)

    def __ge__(self, other):
        return self.values.__ge__(other)

    def __ne__(self, other):
        return self.values.__ne__(other)

    def __lt__(self, other):
        return self.values.__lt__(other)

    def __gt__(self, other):
        return self.values.__gt__(other)

    def __le__(self, other):
        return self.values.__le__(other)

    def _key(self, key):
        """
        Get the data of a hierarchical key
        @param key:
        @return:
        """
        return self.__getitem__((key, ))

    def _loc(self, ind):
        return self.slice_index(ind)

    def _iloc(self, ind):
        ind = slice_to_index(ind, sliced=self.index)
        return self.slice_index(ind)

    def slice_data(self, index):

        if type(index) is not tuple:
            index = (index,)
        index = tuple([slice(None), slice(None), *index])

        if not self.is_cached:
            raise LookupError(f"Cannot slice as data is not cached")

        if self.orientation ==  'simple':
            data = self.data.__getitem(index)

        elif self.orientation == 'index':
            data = recursive_slice(self.data, index)

        else:
            raise LookupError(f"Cannot slice by columns as data is not in simple or index orientation")

        if self.quick_getitem:
            return BeamData.data_batch(data=data, index=self.index, label=self.label)

        return self.clone(data=data, index=self.index, label=self.label, orientation=self.orientation,
                          schema=self.schema, key_fold_map=self.key_fold_map)

    def slice_columns(self, columns):

        if not self.is_cached:
            raise LookupError(f"Cannot slice by columns as data is not cached")

        if self.orientation == 'simple':

            if hasattr(self.data, 'loc'):
                data = self.data[columns]
            else:
                data = self.data[:, self.inverse_columns_map(columns)]

        elif self.orientation == 'index':
            data = recursive_slice_columns(self.data, columns, self.inverse_columns_map(columns))

        else:
            raise LookupError(f"Cannot slice by columns as data is not in simple or index orientation")

        if self.quick_getitem:
            return BeamData.data_batch(data=data, index=self.index, label=self.label)

        return self.clone(data=data, columns=columns, index=self.index, label=self.label,
                          orientation=self.orientation, key_fold_map=self.key_fold_map)

    @property
    def stacked_values(self):

        if not self.is_cached:
            bd = self.cache(in_place=True)
        else:
            bd = self

        if bd.orientation == 'packed':
            raise ValueError("Cannot stack packed data")
        elif bd.orientation == 'simple':
            return bd.data

        data = bd.concatenate_values()
        return data

    @property
    def stacked_index(self):
        index = self.info.index
        return index

    @property
    def stacked_labels(self):
        if 'label' in self.info:
            return self.info.label.values
        return None

    @classmethod
    def simple(cls, *args, preferred_orientation='index', **kwargs):
        bd = cls(*args, preferred_orientation=preferred_orientation, **kwargs)
        return bd.simplified

    @property
    def simplified(self):

        if self.orientation == 'packed':
            raise ValueError("Cannot simplify packed data")
        elif self.orientation == 'simple':
            return self
        elif self.orientation == 'index':
            dim = 0
            orientation = 'simple'
            key_map = self.key_map
        elif self.orientation == 'columns':
            dim = 1
            orientation = 'simple'
            key_map = None
        else:
            raise ValueError("Unknown orientation")

        info = None
        index = None
        label = None
        data = collate_chunks(*self.flatten_data, dim=dim)
        if self.has_index:
            index = collate_chunks(*recursive_flatten(self.index), dim=dim)
        if self.has_label:
            label = collate_chunks(*recursive_flatten(self.label), dim=dim)
        if self._info is not None:
            info = self.info

        return self.clone(data=data, index=index, label=label, info=info, orientation=orientation, key_fold_map=key_map)

    @property
    def stack(self):

        if self.orientation == 'simple':
            return self
        if self.orientation == 'packed':
            raise LookupError(f"Cannot stack for packed orientation")

        data = self.concatenate_values()

        index = self.index
        label = self.label
        if self.orientation == 'index':

            if self.has_index:
                index = self.concatenate_values(recursive_flatten(index), orientation=self.orientation)
            if self.has_label:
                label = self.concatenate_values(recursive_flatten(label), orientation=self.orientation)

        if self.quick_getitem:
            return BeamData.data_batch(data=data, index=index, label=label)

        return self.clone(data=data, index=index, label=label, key_fold_map=None)

    def recursive_filter(self, x, info):

        if info is None:
            return None

        inf_g = info.groupby('fold')
        folds = set(info['fold'].unique())

        def _recursive_filter(xi, flat_key=0):

            if is_container(xi):

                keys = []
                values = []
                index = []
                label = []

                for k, v in iter_container(xi):
                    i, v, l, flat_key = _recursive_filter(v, flat_key=flat_key)
                    if v is not None:
                        values.append(v)
                        index.append(i)
                        keys.append(k)
                        label.append(l)

                argsort, isarange = is_arange(keys)
                if not isarange:
                    values = dict(zip(keys, values))
                    index = dict(zip(keys, index))
                    label = dict(zip(keys, label))
                else:
                    values = [values[j] for j in argsort]
                    index = [index[j] for j in argsort]
                    label = [label[j] for j in argsort]

                return index, values, label, flat_key

            else:

                if flat_key not in folds:
                    return None, None, None, flat_key + 1

                info_in_fold = inf_g.get_group(flat_key)

                in_fold_index = info_in_fold['fold_index']
                x_type = check_type(xi)

                label = info_in_fold['label'] if 'label' in info_in_fold else None
                index = info_in_fold.index

                if xi is None:
                    return None, None, None, flat_key + 1
                elif x_type.minor == Types.native:
                    return index, [xi], label, flat_key + 1
                else:
                    xi_slicer = Slicer(xi, x_type=x_type, wrap_object=True)
                    return index, xi_slicer[in_fold_index.values], label, flat_key + 1

        i, d, l, _ = _recursive_filter(x)

        if all([li is None for li in recursive_flatten(l)]):
            l = None

        return DataBatch(data=d, index=i, label=l)

    def slice_index(self, index, index_type=None):

        if index_type is None:
            index_type = check_type(index, minor=False, element=False)
        if index_type.major == Types.scalar:
            index = [index]
        #     squeeze = True
        # else:
        #     squeeze = False

        if not self.is_cached:
            raise LookupError(f"Cannot slice by index as data is not cached")

        orientation = self.orientation
        if self.orientation in ['simple', 'columns']:

            info = None
            key_fold_map = self.key_fold_map
            if self.has_label:
                if hasattr(self.label, 'loc'):
                    label = self.label.loc[index]
                else:
                    label = self.label[index]
            else:
                label = None

            if self.has_index:
                iloc = self.info['map'].loc[as_numpy(index)].values
            else:
                iloc = index

            if self.orientation == 'simple':
                data = self.data_slicer[iloc]
            else:
                data = recursive_batch(self.data, iloc)

        elif self.orientation in ['index', 'packed']:

            info = None
            key_fold_map = None
            batch_info = self.info.loc[index]
            # batch_info['map'] = np.arange(len(batch_info))

            db = self.recursive_filter(self.data, batch_info)
            data = db.data
            index = db.index
            label = db.label
            if self.orientation == 'index' or batch_info['fold'].nunique() == 1:
                data = collate_chunks(*recursive_flatten(data), dim=0)
                index = collate_chunks(*recursive_flatten(index), dim=0)
                label = collate_chunks(*recursive_flatten(label), dim=0)

                index_map = pd.Series(np.arange(len(index)), index=index)
                index_map = index_map.loc[batch_info.index].values

                # data = Slicer(data)[index_map]
                data = recursive_batch(data, index_map)

                if label is not None:
                    label = label.values[index_map]

                index = batch_info.index
                orientation = 'simple'

                if self.quick_getitem:
                    # if squeeze:
                    #     data = recursive_squeeze(data)
                    #     label = recursive_squeeze(label)
                    #     index = recursive_squeeze(index)
                    return DataBatch(data=data, index=index, label=label)

        else:
            raise ValueError(f"Cannot fetch batch for BeamData with orientation={self.orientation}")

        # if squeeze:
        #     data = recursive_squeeze(data)
        #     label = recursive_squeeze(label)
        #     index = recursive_squeeze(index)

        if self.quick_getitem:
            return BeamData.data_batch(data, index=index, label=label, orientation=self.orientation, info=info,
                                       flatten_index=True, flatten_label=True)

        return self.clone(data=data, columns=self.columns, index=index, label=label,
                          orientation=orientation, info=info, key_fold_map=key_fold_map)

    @staticmethod
    def data_batch(data, index=None, label=None, orientation=None, info=None,
                   flatten_index=False, flatten_label=False):

        ic = is_container(data)
        if ic and len(data) == 1:
            if isinstance(data, dict):
                key = list(data.keys())[0]
            else:
                key = 0

            data = data[key]
            if not flatten_index:
                if index is not None:
                    index = index[key]
                    if isinstance(index, pd.Series):
                        index = index.values
            if not flatten_label:
                if label is not None:
                    label = label[key]
                    if isinstance(label, pd.Series):
                        label = label.values

        elif ic:
            data = BeamData._concatenate_values(data=data, orientation=orientation)

            if index is not None:
                index = BeamData._concatenate_values(data=index, orientation=orientation)

                if info is not None:
                    flat_index = pd.Series(np.arange(len(info)), index=index)
                    inverse_map = flat_index.loc[info.index].values

            if label is not None:
                label = BeamData._concatenate_values(data=label, orientation=orientation)

            if index is not None and info is not None:

                data = data[inverse_map]
                index = info.index.values

                if label is not None:

                    if isinstance(label, pd.Series):
                        label = label.values
                    label = label[inverse_map]

        return DataBatch(data=data, index=index, label=label)

    @staticmethod
    def slice_scalar_or_list(data, keys, data_type=None, keys_type=None, replace_missing=False):

        if data is None:
            return None

        if data_type is None:
            data_type = check_type(data)

        if keys_type is None:
            keys_type = check_type(keys)

        if keys_type.major == Types.scalar:
            if replace_missing:
                if keys not in data:
                    return None
            return data[keys]
        elif keys_type.minor == Types.tuple:
            if replace_missing:
                return get_closest_item_with_tuple_key(data, keys)
            return get_item_with_tuple_key(data, keys)
        else:
            sliced = [] if data_type.minor == Types.list else {}
            for k in keys:
                if replace_missing:
                    if keys not in data:
                        sliced[k] = None
                        continue
                sliced[k] = data[k]
            return sliced

    def get_index_by_key_fold_map(self, keys, keys_type=None):

        if keys_type is None:
            keys_type = check_type(keys)

        if keys_type.major == Types.scalar:
            keys = [keys]

        ind = [self.info['map'].loc[self.info['fold'] == self.key_fold_map[k]].values for k in keys]
        ind = np.concatenate(ind)

        return ind

    @staticmethod
    def update_hierarchy(root_path, all_paths):
        @recursive
        def reduce_hierarchy(x):
            if isinstance(x, list):
                return ['/'.join(xi.split('/')[1:]) for xi in x]
            return '/'.join(x.split('/')[1:])

        while True:
            flat_paths = recursive_flatten(all_paths, flat_array=True)
            if len(flat_paths) == 1:
                return root_path.joinpath(flat_paths[0]), None

            h = flat_paths[0].split('/')[0]
            if all([h == p.split('/')[0] for p in flat_paths]):
                root_path = root_path.joinpath(h)
                all_paths = reduce_hierarchy(all_paths)
            else:
                break

        return root_path, all_paths

    def slice_keys(self, keys):

        data = None
        all_paths = None
        root_path = self.root_path
        keys_type = check_type(keys)
        schema_type = check_type(self.schema)

        if schema_type.major == Types.container and not self.quick_getitem:
            schema = BeamData.slice_scalar_or_list(self.schema, keys, keys_type=keys_type,
                                                   data_type=schema_type, replace_missing=True)
        else:
            schema = self.schema

        if self.is_stored:

            try:
                all_paths = BeamData.slice_scalar_or_list(self.all_paths, keys, keys_type=keys_type,
                                                          data_type=self.data_type)
                root_path, all_paths = BeamData.update_hierarchy(root_path, all_paths)

            except KeyError:
                raise KeyError(f"Cannot find keys: {keys} in stored BeamData object. "
                               f"If the object is archived you should cache it before slicing.")

        if self.is_cached:
            data = BeamData.slice_scalar_or_list(self.data, keys, keys_type=keys_type, data_type=self.data_type)

        if not self.lazy and self.is_stored and data is None:

            data = BeamData.read(self.abs_all_paths(all_paths), schema=schema)

        index = self.index
        label = self.label
        info = self.info
        key_fold_map = self.key_fold_map

        if self.orientation in ['index', 'packed']:
            if self.has_index:
                index = BeamData.slice_scalar_or_list(index, keys, keys_type=keys_type, data_type=self.data_type)
            if self.has_label:
                label = BeamData.slice_scalar_or_list(label, keys, keys_type=keys_type, data_type=self.data_type)
            info = None
            key_fold_map = None

        if self.quick_getitem and data is not None:
            return BeamData.data_batch(data, index=index, label=label, orientation=self.orientation)

        # determining orientation and info can be ambiguous, so we let BeamData to calculate it
        # from the index and label arguments

        # if self.orientation != 'columns' and info is not None:
        #         info = info.loc[index]
        # return BeamData(data=data, path=all_paths, lazy=self.lazy, columns=self.columns,
        #                 index=index, label=label, orientation=self.orientation, info=info)

        return self.clone(data=data, path=root_path, all_paths=all_paths, columns=self.columns, index=index,
                          label=label, schema=schema, info=info, key_fold_map=key_fold_map)

    def clone(self, *args, data=None, path=None, all_paths=None, key_map=None, index=None, label=None,
              columns=None, schema=None, orientation=None,info=None, constructor=None,
              key_fold_map=None, **kwargs):

        name = kwargs.pop('name', self.name)
        lazy = kwargs.pop('lazy', self.lazy)
        device = kwargs.pop('device', self.device)
        target_device = kwargs.pop('target_device', self.target_device)
        override = kwargs.pop('override', self.override)
        compress = kwargs.pop('compress', self.compress)
        split_by = kwargs.pop('split_by', self.split_by)
        chunksize = kwargs.pop('chunksize', self.chunksize)
        chunklen = kwargs.pop('chunklen', self.chunklen)
        n_chunks = kwargs.pop('n_chunks', self.n_chunks)
        partition = kwargs.pop('partition', self.partition)
        archive_size = kwargs.pop('archive_size', self.archive_size)
        preferred_orientation = kwargs.pop('preferred_orientation', self.preferred_orientation)
        read_kwargs = kwargs.pop('read_kwargs', self.read_kwargs)
        write_kwargs = kwargs.pop('write_kwargs', self.write_kwargs)
        quick_getitem = kwargs.pop('quick_getitem', self.quick_getitem)
        glob_filter = kwargs.pop('glob_filter', self.glob_filter)
        write_metadata = kwargs.pop('write_metadata', self.write_metadata)
        read_metadata = kwargs.pop('read_metadata', self.read_metadata)
        metadata_path_prefix = kwargs.pop('metadata_path_prefix', self.metadata_path_prefix)
        chunksize_policy = kwargs.pop('chunksize_policy', self.chunksize_policy)
        # key_fold_map = kwargs.pop('key_fold_map', self.key_fold_map)

        if constructor is None:
            constructor = BeamData

        return constructor(*args, data=data, path=path, name=name, all_paths=all_paths, key_map=key_map,
                 index=index, label=label, columns=columns, lazy=lazy, device=device, target_device=target_device,
                 override=override, compress=compress, split_by=split_by, chunksize=chunksize,
                 chunklen=chunklen, n_chunks=n_chunks, partition=partition, archive_size=archive_size, schema=schema,
                 preferred_orientation=preferred_orientation, read_kwargs=read_kwargs, write_kwargs=write_kwargs,
                 quick_getitem=quick_getitem, orientation=orientation, glob_filter=glob_filter, info=info,
                 chunksize_policy=chunksize_policy, write_metadata=write_metadata, read_metadata=read_metadata,
                 metadata_path_prefix=metadata_path_prefix, key_fold_map=key_fold_map, **kwargs)

    def inverse_columns_map(self, columns):

        columns_map = self.columns_map
        if check_type(columns).major == Types.scalar:
            columns = columns_map[columns]
        else:
            columns = [columns_map[i] for i in columns]

        return columns

    def __repr__(self):
        if self.is_cached and self.orientation == 'simple':
            return repr(self.data)
        return self.__str__()

    def update_all_paths_file(self):

        path = self.metadata_paths['all_paths']
        if self.write_metadata:
            BeamData.write_file(self.all_paths, path)

    def __str__(self):

        if self.is_cached and self.orientation == 'simple':
            return f"BeamData (simple): {self.name}\n{self.data}"

        params = {'orientation': self.orientation, 'lazy': self.lazy, 'stored': self.is_stored,
                  'cached': self.is_cached, 'device': self.device, 'objects_type': self.objects_type,
                  'quick_getitem': self.quick_getitem, 'has_index': self.has_index,
                  'has_label': self.has_label}
        params_line = ' | '.join([f"{k}: {v}" for k, v in params.items()])

        s = f"BeamData: {self.name}\n"
        s += f"  path: \n"
        s += f"  {self.root_path} \n"
        s += f"  params: \n"
        s += f"  {params_line} \n"
        s += f"  keys: \n"
        s += f"  {self.hierarchical_keys()} \n"
        s += f"  sizes:\n"
        s += f"  {self.size} \n"
        s += f"  shapes:\n"
        s += f"  {self.shape} \n"
        s += f"  types:\n"
        s += f"  {self.dtypes} \n"
        return s

    @property
    def hash(self):
        return beam_hash(DataBatch(index=self.index, label=self.label, data=self.data))

    @staticmethod
    def set_first_key(key, value):
        data = {}
        if type(key) is tuple:
            for k in key[:-1]:
                data[k] = {}
                data = data[k]
            data[key[-1]] = value
        else:
            data[key] = value
        return data

    def clear_index(self):
        self._index = None
        self._has_index = None

    def __setitem__(self, key, value):
        """
        Set value supports only key hierarchy except for the case of 'simple' orientation.
        @param key:
        @param value:
        """

        if self.synced:
            self.is_stored = True
            self.is_cached = True

        org_key = key
        key_type = check_type(key)

        if self.is_stored:

            kwargs = self.get_default_params('compress', 'chunksize', 'chunklen', 'n_chunks', 'partition',
                                              'split_by', 'archive_size', 'override')

            path = self.root_path
            all_paths = self.all_paths
            if type(all_paths) is str:
                all_paths = {BeamData.default_data_file_name: all_paths}
            elif self.all_paths is None:
                all_paths = {}

            if key_type.major != Types.scalar:

                for i, k in enumerate(key[:-1]):

                    if type(all_paths) is dict:
                        assert type(k) is str, f"key {k} is not a string"
                    if type(all_paths) is list:
                        assert type(k) is int and k <= len(all_paths), f"key {k} is not an integer"

                    if k not in all_paths:
                        if type(key[i + 1]) is int:
                            all_paths[k] = []
                        else:
                            all_paths[k] = {}

                    all_paths = all_paths[k]
                    path = path.joinpath(BeamData.normalize_key(k))

                key = key[-1]

            path = path.joinpath(key)
            all_paths[key] = BeamData.write_object(value, path, **kwargs)
            self._all_paths = all_paths
            self.update_all_paths_file()

        if self.is_cached:

            key = org_key
            if self.orientation == 'simple':
                self.data.__setitem__(key, value)
            else:
                if self.data is None:
                    self.data = self.set_first_key(key, value)
                else:
                    set_item_with_tuple_key(self.data, key, value)

            if self.orientation == 'index':
                if self.has_index:
                    logger.warning("Previous index value conflicts with new item. Setting index to None.")
                self.clear_index()
                if self.has_label:
                    logger.warning("Previous label value conflicts with new item. Setting index to None.")

            self.clear_label()
            self.reset_metadata('_all_paths')

        if not self.is_stored and self.data is None:
            self.data = self.set_first_key(key, value)
            self.reset_metadata('_all_paths')
            self.is_cached = True

    def apply(self, func, *args, preferred_orientation='columns', **kwargs):
        data = recursive(func)(self.data,  *args, **kwargs)
        return self.clone(data, index=self.index, label=self.label, info=self.info, key_fold_map=self.key_fold_map,
                          preferred_orientation=preferred_orientation)

    def clear_label(self):
        self._label = None
        self._has_label = None

    def reset_index(self):
        return self.clone(self.data, index=None, label=self.label, schema=self.schema)

    @staticmethod
    def get_n_chunks(data, n_chunks=None, chunklen=None, chunksize=None, size=None, chunksize_policy='round'):

        if chunksize_policy == 'round':
            round_func = np.round
        elif chunksize_policy == 'ceil':
            round_func = np.ceil
        elif chunksize_policy == 'floor':
            round_func = np.floor
        else:
            raise ValueError(f"Unsupported chunksize_policy: {chunksize_policy}")

        if (n_chunks is None) and (chunklen is None):
            if size is None:
                size = sum(recursive_flatten(recursive_size(data), flat_array=True))
            n_chunks = max(int(round_func(size / chunksize)), 1)
        elif (n_chunks is not None) and (chunklen is not None):
            logger.warning("splitting to chunks requires only one of chunklen|n_chunks. Defaults to using n_chunks")
        elif n_chunks is None:
            n_chunks = max(int(np.round(container_len(data) / chunklen)), 1)

        return n_chunks

    @property
    def schema_type(self):
        if self._schema_type is None:
            self._schema_type = check_type(self.schema)
        return self._schema_type

    def divide_chunks(self, **kwargs):

        split_by = kwargs.pop('split_by', self.split_by)
        chunksize = kwargs.pop('chunksize', self.chunksize)
        chunklen = kwargs.pop('chunklen', self.chunklen)
        n_chunks = kwargs.pop('n_chunks', self.n_chunks)
        partition = kwargs.pop('partition', self.partition)
        chunksize_policy = kwargs.pop('chunksize_policy', self.chunksize_policy)

        if not self.is_cached and split_by != 'keys':

            if not self.lazy:
                self.cache()
            else:
                raise ValueError(f"split_by={split_by} is not supported for not-cached and lazy data.")

        if split_by == 'keys':

            if self.is_cached:

                for i, (k, d) in enumerate(self.flatten_items.items()):

                    s = BeamData.get_schema_from_tupled_key(self.schema, k)
                    index = None
                    if self.has_index:
                        index = get_item_with_tuple_key(self.index, k)
                    label = None
                    if self.has_label:
                        label = get_item_with_tuple_key(self.label, k)

                    info = None
                    if self.info is not None:
                        info = self.info[self.info['fold_index'] == i]
                    yield k, self.clone(d, index=index, label=label, schema=s, info=info)

            else:

                for i, (k, p) in enumerate(recursive_flatten_with_keys(self.all_paths).items()):
                    s = get_item_with_tuple_key(self.schema, k)

                    info = None
                    if self.info is not None:
                        info = self.info[self.info['fold_index'] == i]

                    yield k, self.clone(path=self.root_path, all_paths={BeamData.default_data_file_name: p}, schema=s, info=info)

        else:

            n_chunks = BeamData.get_n_chunks(self.data, n_chunks=n_chunks, chunklen=chunklen, chunksize=chunksize,
                                             size=self.total_size, chunksize_policy=chunksize_policy)

            if split_by == 'column':
                dim = 1
                for k, data_i in recursive_chunks(self.data, n_chunks, dim=dim):
                    if self.quick_getitem:
                        yield k, BeamData.data_batch(data_i, index=self.index, label=self.label)
                    else:
                        yield k, self.clone(data_i, index=self.index, label=self.label)

            elif split_by == 'index':
                dim = 0
                for k, data in recursive_chunks((self.index, self.data, self.label), n_chunks=n_chunks,
                                                dim=dim, partition=partition):
                    index_i, data_i, label_i = data

                    if self.quick_getitem:
                        yield k, BeamData.data_batch(data_i, index=index_i, label=label_i)
                    else:
                        yield k, self.clone(data_i, index=index_i, label=label_i)

            else:
                raise ValueError(f"split_by={split_by} is not supported.")

    def __iter__(self):

        for v in self.divide_chunks():
            yield v

    def sample(self, n, replace=True):

        if replace:
            ind = np.random.choice(len(self), size=n, replace=True)
        else:
            ind = np.random.randint(len(self), size=n)

        ind = self.info.loc[ind].index
        return self[ind]

    def head(self, n=20):

        ind = np.arange(min(len(self), n))

        ind = self.info.loc[ind].index
        return self[ind]

    def __getitem__(self, item):

        '''

        @param item:
        @return:

        The axes of BeamData objects are considered to be in order: [keys, index, columns, <rest of shape>]
        if BeamData is orient==simple (meaning there are no keys), the first axis disappears.

        Optional item configuration:


        [keys] - keys is ether a slice, list or scalar.
        [index] - index is pandas/numpy/tensor array
        [keys, index] - keys is ether a slice, list or scalar and index is an array

        '''

        if self.orientation == 'simple' and self.key_fold_map is None:
            axes = ['index', 'columns', 'else']
            short_list = True
        else:
            axes = ['keys', 'index', 'columns', 'else']
            short_list = False

        obj = self
        if type(item) is not tuple:
            item = (item, )
        for i, ind_i in enumerate(item):

            # skip if this is a full slice
            if type(ind_i) is slice and ind_i == slice(None):
                axes.pop(0)
                continue

            i_type = check_type(ind_i)
            # skip the first axis in these case
            if axes[0] == 'keys' and (i_type.minor in [Types.pandas, Types.numpy, Types.slice, Types.tensor, Types.cudf]):
                axes.pop(0)
            if axes[0] == 'keys' and (i_type.minor == Types.list and i_type.element == Types.int):
                axes.pop(0)
            if (axes[0] == 'keys' and (i_type.major == Types.scalar and i_type.element == Types.int)
                    and check_type(list(self.hierarchical_keys()), minor=False).element == Types.str):
                axes.pop(0)
            # for orientation == 'simple' we skip the first axis if we slice over columns and index_type is not str
            if (short_list and axes[0] == 'index' and i_type.element == Types.str and
                    self.index_type.element != Types.str):
                axes.pop(0)

            a = axes.pop(0)
            if a == 'keys' and self.key_fold_map is None:
                obj = obj.slice_keys(ind_i)

            elif a == 'keys' and self.key_fold_map is not None:

                ind_i = self.get_index_by_key_fold_map(ind_i, keys_type=i_type)
                obj = obj.slice_index(ind_i)

            elif a == 'index':

                if i_type.major == Types.slice:
                    ind_i = slice_to_index(ind_i, l=len(obj))

                if i_type.element == Types.bool:
                    ind_i = self.info.iloc[ind_i].index

                if not isinstance(obj, BeamData):
                    ValueError(f"quick_getitem supports only a single index slice")

                obj = obj.slice_index(ind_i)

            elif a == 'columns':

                if not isinstance(obj, BeamData):
                    ValueError(f"quick_getitem supports only a single index slice")

                obj = obj.slice_columns(ind_i)

            else:

                ind_i = item[i:]
                obj = obj.slice_data(ind_i)
                break

        return obj
