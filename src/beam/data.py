import itertools
import numpy as np
import torch
from .utils import check_type, slice_to_index, as_tensor, to_device, recursive_batch, as_numpy, beam_device, \
    recursive_device, container_len, recursive, recursive_len, recursive_shape, recursive_types, retrieve_name
import pandas as pd
import math
from collections import namedtuple
from .utils import divide_chunks, collate_chunks, recursive_chunks, iter_container, logger, \
    recursive_size_summary, container_len, is_arange, is_chunk, \
    recursive_size, recursive_flatten, recursive_collate_chunks, recursive_keys, recursive_slice_columns, \
    recursive_slice, recursive_flatten_with_keys, get_item_with_tuple_key, PureBeamPath, set_item_with_tuple_key, \
    get_closest_item_with_tuple_key
import os
from .path import BeamPath, beam_path
from functools import partial
from collections import defaultdict
import time

DataBatch = namedtuple("DataBatch", "index label data")


class Groups(object):

    def __init__(self, groupby_pointer):
        self.groupby_pointer = groupby_pointer
        self.groups = {}

    def __getitem__(self, ind):

        if ind not in self.groups:
            self.groups[ind] = self.groupby_pointer().get_group(ind)

        return self.groups[ind]


class Iloc(object):

    def __init__(self, pointer):
        self.pointer = pointer

    def __getitem__(self, ind):
        return self.pointer._iloc(ind)


class Loc(object):

    def __init__(self, pointer):
        self.pointer = pointer

    def __getitem__(self, ind):
        return self.pointer._loc(ind)


class Key(object):

    def __init__(self, pointer):
        self.pointer = pointer

    def __getitem__(self, ind):
        return self.pointer._key(ind)


class BeamSchema(object):

    def __init__(self, read_schema=None, write_schema=None,  **kwargs):
        self._schema = {}
        self._schema.update(kwargs)
        self._read_schema = read_schema or {}
        self._write_schema = write_schema or {}

    def __getitem__(self, item):

        if item in self._schema.keys():
            return self._schema[item]
        elif item in self._read_schema.keys():
            return self._read_schema[item]
        elif item in self._write_schema.keys():
            return self._write_schema[item]
        else:
            raise KeyError(f'No schema for {item}')

    @property
    def read_schema(self):
        return {**self._schema, **self._read_schema}

    @property
    def write_schema(self):
        return {**self._schema, **self._write_schema}


class BeamData(object):

    # metadata files
    metadata_files = {'conf': '.conf.pkl', 'schema': '.schema.pkl',
                      'info': '.info.fea', 'label': '.label', 'aux': '.aux.pkl',
                      'index': '.index', 'all_paths': '.all_paths.pkl'}

    default_data_file_name = 'data'
    index_chunk_file_extension = '.index_chunk'
    columns_chunk_file_extension = '.columns_chunk'
    index_partition_directory_name = '.index_part'
    columns_partition_directory_name = '.columns_part'

    def __init__(self, *args, data=None, path=None, name=None, all_paths=None,
                 index=None, label=None, columns=None, lazy=True, device=None, target_device=None, schema=None,
                 override=True, compress=None, split_by='keys', chunksize=int(1e9), chunklen=None, n_chunks=None,
                 key_map=None, partition=None, archive_size=int(1e6), preferred_orientation='columns', read_kwargs=None,
                 write_kwargs=None, quick_getitem=False, orientation=None, glob_filter=None, info=None, **kwargs):

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
         different length but their shape[1:] is identical so we can collect batch of elements and concat them together.

         4. packed: each data element represents different set of data points but each data point may have different nature.
         this could model for example node properties in Knowledge graph where there are many types of nodes with different
         features. In case there is a common index, it can be used to slice and collect the data like the original
         PackedFold object.

         5. simplified_index: a simple orientation that was generated from a collated version of index orientation.

         If data is both cached in self.data and stored in self.all_paths, the cached version is always preferred.

        The orientation is inferred from the data. If all obje'data'cts have same length they are assumed to represent columns
        orientation. If all objects have same shape[1:] they are assumed to represent index orientation. If one wish to pass
        an index orientation data where all objects have same length, one can pass the preferred_orientation='index' argument.

        '''

        #todo: add support for target device+to_tensor when returning DataBatch

        self.lazy = lazy
        self.override = override
        self.compress = compress
        self.chunksize = chunksize
        self.chunklen = chunklen
        self.n_chunks = n_chunks
        self.partition = partition
        self.archive_size = archive_size
        self.target_device = target_device
        self._index = index
        self._label = label
        self._schema = schema
        self.columns = columns
        self.preferred_orientation = preferred_orientation
        self.split_by = split_by
        self.quick_getitem = quick_getitem
        self.glob_filter = glob_filter
        self._key_map = key_map

        self.stored = False
        self.cached = True

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
            self.cached = False
            # in this case the data is not cached, so it should be stored ether in root_path or in all_paths

        if device is not None:
            self.as_tensor(device=device)

        path = beam_path(path)
        path_type = check_type(path)

        self._name = name

        if path is not None:
            self._name = path.name
        else:
            self._name = None

        if path_type.major == 'container':
            self._all_paths = path
        elif path is not None:

            self._root_path, self._all_paths, self._metadata_paths = BeamData.single_file_case(path, all_paths,
                                                                                               self._metadata_paths)
            self._root_path = path

        if ((self._all_paths is not None) or (self._root_path is not None and self._root_path.not_empty())) and \
                not self.cached:
            self.stored = True
            if not lazy:
                self.cache()

    @property
    def metadata_paths(self):

        if self._metadata_paths is not None:
            return self._metadata_paths
        self._metadata_paths = {k: self.root_path.joinpath(v) for k, v in BeamData.metadata_files.items()}

        return self._metadata_paths

    @property
    def root_path(self):
        if self._root_path is not None:
            return self._root_path

        if self.all_paths is not None:
            self._root_path = self.recursive_root_finder(self.all_paths)
            return self._root_path

        return self._root_path

    @property
    def all_paths(self):

        if self._all_paths is not None:
            return self._all_paths

        if self.stored:
            path = self.metadata_paths['all_paths']
            if path.exists():
                self._all_paths = path.read()

            else:
                self._all_paths = BeamData.recursive_map_path(self.root_path, glob_filter=self.glob_filter)
                BeamData.write_file(self._all_paths, path)

        return self._all_paths

    def get_info_groups(self):
        if self._info_groups is not None:
            return self._info_groups
        self._info_groupby = self.info.groupby('fold')
        return self._info_groupby

    @property
    def index(self):
        if self._index is not None:
            return self._index

        if self.stored:

            for path in self.metadata_paths['index'].parent.iterdir():
                if path.stem == BeamData.metadata_files['index']:
                    if path.exists():
                        self._index = path.read()
                        return self._index

        info = self.info
        if self.orientation in ['columns', 'simple',  'simplified_index']:
            self._index = info.index.values
        else:
            # TODO: support index for packed and index case
            self._index = None
            # info_filtered = BeamData.recursive_filter(self.size, self.info)
            # if info_filtered is not None:
            #     self._index = info_filtered.index
            #     if self._objects_type == 'tensor':
            #
            #         func = partial(as_tensor, device=self.device, dtype=None, return_vector=False)
            #         self._index = recursive(func)(self._index)

        return self._index

    @property
    def label(self):

        if self._label is not None:
            return self._label

        if self.stored:

            for path in self.metadata_paths['label'].parent.iterdir():
                if path.stem == BeamData.metadata_files['label']:
                    if path.exists():
                        self._label = path.read()
                        return self._label

        return self._label

    @staticmethod
    def single_file_case(root_path, all_paths, metadata_paths):

        if root_path.parent.is_dir():
            for p in root_path.parent.iterdir():
                if p.stem == root_path.stem and p.is_file():

                    meta_path = p.parent.joinpath(f'.{p.name}')
                    meta_path.mkdir()
                    metadata_paths = {k: meta_path.joinpath(v) for k, v in BeamData.metadata_files.items()}
                    root_path = p
                    all_paths = {'data': ''}
                    break

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

        return bd.clone(data, columns=columns, index=index, label=label)

    @classmethod
    def from_path(cls, path, *args, **kwargs):
        return cls(path=path, *args, **kwargs)

    @classmethod
    def from_indexed_pandas(cls, data, *args, **kwargs):

        @recursive
        def get_index(x):
            return x.index

        index = get_index(data)
        kwargs['index'] = index

        return cls(data, *args, **kwargs)

    @property
    def name(self):
        if self._name is None:
            self._name = retrieve_name(self)
        return self._name

    @property
    def objects_type(self):

        if self._objects_type is not None:
            return self._objects_type

        objects_types = recursive_flatten(self.data_types)
        objects_types = [v.minor for v in objects_types if v.minor != 'none']

        u = np.unique(objects_types)

        if len(u) == 1:
            self._objects_type = u[0]
        else:
            self._objects_type = 'mixed'

        return self._objects_type

    @property
    def schema(self):

        if self._schema is not None:
            return self._schema

        if self.stored:
            schema_path = self.metadata_paths['schema']
            if schema_path.is_file():
                self._schema = schema_path.read()
                return self._schema

        return self._schema

    @property
    def conf(self):

        if self._conf is not None:
            return self._conf

        if self.stored:
            conf_path = self.metadata_paths['conf']
            if conf_path.is_file():
                self._conf = conf_path.read()
                return self._conf

        if self.cached:
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
        return defaultdict(lambda: None)

    @property
    def key_map(self):
        if self._key_map is not None:
            return self._key_map
        self._key_map = {k: i for i, (k, v) in enumerate(self.flatten_items.items())}

        return self._key_map

    @property
    def info(self):

        if self._info is not None:
            return self._info

        if self.stored:
            info_path = self.metadata_paths['info']
            if info_path.is_file():
                self._info = info_path.read()
                return self._info

        if self.cached:

            if self.orientation in ['index', 'packed']:
                filtered_data = list(filter(lambda x: x is not None, self.flatten_data))
                if not len(filtered_data):
                    self._info = None
                    return self._info
                fold_index = np.concatenate([np.arange(len(d)) for d in filtered_data])
                fold = np.concatenate([np.full(len(d), k) for k, d in enumerate(filtered_data)])

                # still not sure if i really need this column. if so, it should be fixed
                # fold_key = np.concatenate([np.full(len(d), k) for k, d in self.flatten_items.items()])
                lengths = np.array([len(d) for d in self.flatten_data])
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
                index = np.arange(len(self))

            if self.label is not None:
                label = np.concatenate([as_numpy(l) for l in recursive_flatten([self.label])])
            else:
                label = None

            info = {'fold': fold, 'fold_index': fold_index,
                    # 'fold_key': fold_key,
                    'offset': offset,
                    'map': np.arange(len(index))}

            if self.label is not None:
                info['label'] = label

            self._info = pd.DataFrame(info, index=index)
            return self._info

        self._info = None
        return self._info

    @property
    def path(self):
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
        self.stored = False

    @property
    def index_mapper(self):

        info = self.info
        if 'map' in info.columns:
            return info['map']

        return None

    @staticmethod
    def normalize_key(key):
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

        if self.objects_type == 'tensor':
            self._device = recursive_device(self.data)
        else:
            self._device = None

        return self._device

    def to(self, device):
        self.data = recursive(lambda x: x.to(device))(self.data)
        self._device = device
        return self

    def __len__(self):

        if self._len is not None:
            return self._len

        if self.stored and self._conf is not None:
            self._len = self._conf['len']
            return self._len

        if self.cached:
            if self.orientation == 'columns':
                self._len = container_len(self.data)
            else:
                self._len = sum(recursive_len(self.flatten_data))
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

        if self.cached:

            data_type = check_type(self.data)

            if data_type.major != 'container':
                self._orientation = 'simple'
                if hasattr(self.data, 'columns') and self.columns is None:
                    self.columns = self.data.columns
                if hasattr(self.data, 'index') and self.index is None:
                    self._index = self.data.index

            else:

                if self.preferred_orientation == 'columns':
                    lens = recursive_flatten(recursive_len([self.data]), flat_array=True)
                    lens = list(filter(lambda x: x is not None, lens))

                    lens_index = recursive_flatten(recursive_len([self._index]), flat_array=True)
                    lens_index = list(filter(lambda x: x is not None, lens_index))

                    if len(np.unique(lens)) == 1 and len(lens_index) <= 1:
                        self._orientation = 'columns'
                        return self._orientation

                shapes = recursive_flatten(recursive(lambda x: tuple(x.shape[1:])
                                                     if hasattr(x, 'shape') else None)([self.data]))

                shapes = list(filter(lambda x: x is not None, shapes))
                if len(np.unique(shapes)) == 1:
                    self._orientation = 'index'
                else:
                    self._orientation = 'packed'

        elif self.stored:
            self._orientation = self.conf['orientation']

        else:
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
        path = path.write(data, **kwargs)

        return path

    @staticmethod
    def get_schema_from_subset(schema, key, schema_type=None):

        if schema_type is None:
            schema_type = check_type(schema)

        if schema_type.minor == 'dict' and key in schema:
            s = schema[key]
        elif schema_type.minor == 'list' and key < len(schema):
            s = schema[key]
        elif schema_type.major == 'container':
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

        if not is_arange(keys):
            values = dict(zip(keys, values))
            # values = {k: values[k] for k in sorted(values.keys())}
        else:
            values = [values[i] for i in np.argsort(keys)]

        if type(values) is dict and 'data' in values and len(values) == 1:
            values = values['data']

        return values

    @staticmethod
    def recursive_root_finder(all_paths, head=None):
        if head is None:
            head = []

        all_paths_type = check_type(all_paths)
        if all_paths_type.major == 'container':

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

            if not is_arange(keys):
                values = dict(zip(keys, values))
            else:
                values = [values[i] for i in np.argsort(keys)]

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
        self._index = func(self.index)
        self._label = func(self.label)
        self._objects_type = 'tensor'

        return self

    def as_numpy(self):

        '''
        Convert the data to numpy in place
        @return:
        '''

        func = partial(as_numpy)
        self.data = recursive(func)(self.data)
        self._index = func(self.index)
        self._label = func(self.label)
        self._objects_type = 'numpy'

        return self

    @property
    def values(self):

        if not self.cached:
            self.cache()

        return self.data

    @staticmethod
    def read(paths, schema=None, **kwargs):

        paths_type = check_type(paths)

        if paths is None:
            return None

        if paths_type.major == 'container':
            keys = []
            values = []

            schema_type = check_type(schema)

            for k, next_path in iter_container(paths):

                s = BeamData.get_schema_from_subset(schema, k, schema_type=schema_type)
                values.append(BeamData.read(next_path, schema=s, **kwargs))
                keys.append(k)

            return BeamData.containerize_keys_and_values(keys, values)

        path = beam_path(paths)
        if schema is not None:
            kwargs = {**schema.read_schema, **kwargs}

        if path.is_file():
            return path.read(**kwargs)

        if path.is_dir():

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

        for p in path.parent.iterdir():
            if p.stem == path.stem:
                return p.read(**kwargs)

        logger.warning(f"No object found in path: {path}")
        return None

    @staticmethod
    def write_tree(data, path, sizes=None, split_by='keys', archive_size=int(1e6), chunksize=int(1e9),
                   chunklen=None, n_chunks=None, partition=None, file_type=None, root=False, schema=None, **kwargs):

        path = beam_path(path)

        if sizes is None:
            sizes = recursive_size(data)

        data_type = check_type(data)

        if data_type.major == 'container':

            size_summary = sum(recursive_flatten(sizes, flat_array=True))

            if size_summary < archive_size:

                if root:
                    path = path.joinpath(BeamData.default_data_file_name)

                BeamData.write_object(data, path, size=size_summary, archive=True, **kwargs)
                return

            schema_type = check_type(schema)
            for k, v in iter_container(data):

                s = BeamData.get_schema_from_subset(schema, k, schema_type=schema_type)
                BeamData.write_tree(v, path.joinpath(BeamData.normalize_key(k)), sizes=sizes[k],
                                    archive_size=archive_size, chunksize=chunksize, chunklen=chunklen,
                                    split_by=split_by, n_chunks=n_chunks, partition=partition, root=False,
                                    file_type=file_type, schema=s, **kwargs)

        else:

            if root:
                path = path.joinpath(BeamData.default_data_file_name)

            BeamData.write_object(data, path, size=sizes, archive=False,
                                        chunksize=chunksize, chunklen=chunklen, split_by=split_by,
                                        n_chunks=n_chunks, partition=partition, schema=schema,
                                        file_type=file_type, **kwargs)

    @staticmethod
    def write_object(data, path, override=True, size=None, archive=False, compress=None, chunksize=int(1e9),
              chunklen=None, n_chunks=None, partition=None, file_type=None, schema=None, split_by=None, **kwargs):

        path = beam_path(path)

        if not override:
            if path.exists() or (path.parent.is_dir() and any(p.stem == path.stem for p in path.parent.iterdir())):
                logger.warning(f"path {path} exists. To override, specify override=True")
                return

        if archive:
            object_path = BeamData.write_file(data, path.with_suffix('.pkl'), override=override,
                                              schema=schema, **kwargs)
        else:

            if split_by != 'keys':
                n_chunks = BeamData.get_n_chunks(data, chunksize=chunksize, chunklen=chunklen,
                                                 n_chunks=n_chunks, size=size)

            data_type = check_type(data)
            if partition is not None and data_type.minor == 'pandas':
                priority = ['.parquet', '.fea', '.pkl']
            elif data_type.minor in ['pandas', 'numpy']:
                priority = ['.fea', '.parquet', '.pkl']
            elif data_type.minor == 'scipy_sparse':
                priority = ['scipy_npz', 'npy', '.pkl']
            elif data_type.minor == 'tensor':
                priority = ['.pt']
            else:
                priority = ['.pkl']

            if file_type is not None:
                priority.insert(file_type, 0)

            if split_by != 'keys' and n_chunks > 1:
                dim = {'index': 0, 'columns': 1}[split_by]
                data = list(divide_chunks(data, n_chunks=n_chunks, dim=dim))
                chunk_file_extension = {'index': BeamData.index_chunk_file_extension,
                                        'columns': BeamData.columns_chunk_file_extension}[split_by]
                object_path = path

            else:
                data = [(0, data), ]
                chunk_file_extension = ''

            for i, di in data:

                if len(data) > 1:
                    path_i = path.joinpath(f"{i:06}{chunk_file_extension}")
                else:
                    path_i = path

                for ext in priority:
                    file_path = path_i.with_suffix(ext)

                    if len(data) == 1:
                        object_path = file_path
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
                            if compress is False:
                                kwargs['compressed'] = True
                            BeamData.write_file(di, file_path, schema=schema, **kwargs)

                        else:
                            BeamData.write_file(di, file_path, schema=schema, **kwargs)

                        error = False
                        priority = [ext]
                        break

                    except Exception as e:
                        logger.warning(f"Failed to write file: {file_path.name}. Trying with the next file extension")
                        logger.debug(e)
                        error = True
                        if file_path.exists():
                            file_path.unlink()

                if error:
                    logger.error(f"Could not write file: {path_i.name}.")

        return object_path
    @property
    def columns_map(self):

        if self._columns_map is not None:
            return self._columns_map

        if self.columns is not None:
            self._columns_map = {str(k): i for i, k in enumerate(self.columns)}

        self._columns_map = None
        return self._columns_map

    def keys(self):
        if self.orientation == 'simple':
            return self.columns
        if self.cached:
            return recursive_keys(self.data)
        return recursive_keys(self.all_paths)

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
    def _concatenate(data, orientation=None, objects_type=None):

        data = recursive_flatten(data)
        if orientation is None:
            orientation = 'index'
        if objects_type is None:
            _, v = next(iter_container(data))
            objects_type = check_type(v).minor

        if orientation == 'columns':
            dim = 1
        elif orientation == 'index':
            dim = 0
        else:
            return data

        if objects_type == 'tensor':
            func = torch.cat
            kwargs = {'dim': dim}
        elif objects_type == 'pandas':
            func = pd.concat
            data = [pd.Series(v.values) if isinstance(v, pd.Index) else v for v in data]
            kwargs = {'axis': dim}
        elif objects_type == 'numpy':
            func = np.concatenate
            kwargs = {'axis': dim}
        else:
            logger.warning(f"Concatenation not implemented for {objects_type}, returning the original data")
            return data

        return func(data, **kwargs)

    def concatenate(self, data=None, orientation=None, objects_type=None):

        if data is None:
            data = self.flatten_data
            orientation = self.orientation
            objects_type = self.objects_type

        return BeamData._concatenate(data, orientation=orientation, objects_type=objects_type)

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

    def store(self, data=None, path=None, compress=None, chunksize=None,
              chunklen=None, n_chunks=None, partition=None, split_by=None,
              archive_size=None, override=True, **kwargs):

        path = self.get_default('root_path', path)

        sizes = None
        if data is None:
            data = self.data
            sizes = self.size

        path.clean()

        kwargs = self.get_default_params(compress=compress, chunksize=chunksize, chunklen=chunklen, n_chunks=n_chunks,
                                         partition=partition, split_by=split_by, archive_size=archive_size,
                                         override=override, **kwargs)

        BeamData.write_tree(data, path, root=True, sizes=sizes, schema=self.schema, **kwargs)

        # store info and conf files
        info_path = path.joinpath(BeamData.metadata_files['info'])
        BeamData.write_object(self.info, info_path)
        conf_path = path.joinpath(BeamData.metadata_files['conf'])
        BeamData.write_object({**self.conf}, conf_path, archive=True)
        conf_path = path.joinpath(BeamData.metadata_files['conf'])
        BeamData.write_object({**self.conf}, conf_path, archive=True)

        # store index and label
        if self.index is not None:
            index_path = path.joinpath(BeamData.metadata_files['index'])
            BeamData.write_object(self.index, index_path)
        if self.label is not None:
            label_path = path.joinpath(BeamData.metadata_files['label'])
            BeamData.write_object(self.label, label_path)

        self.stored = True
        self._root_path = path
        self.data = data

        self._all_paths = BeamData.recursive_map_path(self.root_path, glob_filter=self.glob_filter)
        path = self.root_path.joinpath(BeamData.metadata_files['all_paths'])
        BeamData.write_file(self._all_paths, path)

    def state_dict(self):
        if not self.cached:
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
            return root_path.joinpath(path)

        if all_paths is None:
            all_paths = self.all_paths

        return _abs_all_paths(all_paths)

    @classmethod
    def load_state_dict(cls, state_dict):
        return cls(**state_dict)

    def cache(self, path=None, all_paths=None, schema=None, update=False, **kwargs):

        if self.cached:
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

        if not self.stored:
            logger.warning("stored=False, data is seems to be un-synchronized")

        data = BeamData.read(self.abs_all_paths(all_paths), schema=schema, **kwargs)

        # if type(data) is dict and 'data' in data and len(data) == 1:
        #     data = data['data']

        self._root_path = path
        # self.all_paths = BeamData.recursive_map_path(root_path, glob_filter=self.glob_filter)
        self._all_paths = all_paths
        self.data = data
        self.stored = True
        self.cached = True

        self.reset_metadata()
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

    def inverse_map(self, ind):

        ind = slice_to_index(ind, l=len(self), sliced=self.index)

        index_type = check_type(ind)
        if index_type.major == 'scalar':
            ind = [ind]

        if self.index_mapper is not None:
            ind = self.index_mapper.loc[ind].values

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
        ind = self.inverse_map(ind)
        return self.slice_index(ind)

    def _iloc(self, ind):

        ind = slice_to_index(ind, l=len(self), sliced=self.index)
        index_type = check_type(ind)
        if index_type.major == 'scalar':
            ind = [ind]

        return self.slice_index(ind)

    def slice_data(self, index):

        if type(index) is not tuple:
            index = (index,)
        index = tuple([slice(None), slice(None), *index])

        if not self.cached:
            raise LookupError(f"Cannot slice as data is not cached")

        if self.orientation in ['simple', 'simplified_index']:
            data = self.data.__getitem(index)

        elif self.orientation == 'index':
            data = recursive_slice(self.data, index)

        else:
            raise LookupError(f"Cannot slice by columns as data is not in simple or index orientation")

        if self.quick_getitem:
            return BeamData.data_batch(data=data, index=self.index, label=self.label)

        return self.clone(data=data, index=self.index, label=self.label, orientation=self.orientation,
                          schema=self.schema)

    def slice_columns(self, columns):

        if not self.cached:
            raise LookupError(f"Cannot slice by columns as data is not cached")

        if self.orientation in ['simple', 'simplified_index']:

            if hasattr(self.data, 'loc'):
                data = self.data[columns]
            else:
                data = self.data[:, columns]

        elif self.orientation == 'index':
            data = recursive_slice_columns(self.data, columns)

        else:
            raise LookupError(f"Cannot slice by columns as data is not in simple or index orientation")

        if self.quick_getitem:
            return BeamData.data_batch(data=data, index=self.index, label=self.label)

        return self.clone(data=data, columns=columns, index=self.index, label=self.label, orientation=self.orientation)

    @property
    def stacked_values(self):
        data = self.concatenate()
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
    def simple(cls, *args, **kwargs):
        bd = cls(*args, **kwargs)
        return bd.simplified

    @property
    def simplified(self):

        if self.orientation == 'packed':
            raise ValueError("Cannot simplify packed data")
        elif self.orientation == 'simple':
            return self
        elif self.orientation == 'index':
            dim = 0
            orientation = 'simplified_index'
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
        if self.index is not None:
            index = collate_chunks(*recursive_flatten(self.index), dim=dim)
        if self.label is not None:
            label = collate_chunks(*recursive_flatten(self.label), dim=dim)
        if self._info is not None:
            info = self.info

        return self.clone(data=data, index=index, label=label, info=info, orientation=orientation, key_map=key_map)

    @property
    def stack(self):

        if self.orientation in ['simple', 'simplified_index']:
            return self
        if self.orientation == 'packed':
            raise LookupError(f"Cannot stack for packed orientation")

        data = self.concatenate()

        index = self.index
        label = self.label
        if self.orientation == 'index':

            if index is not None:
                index = self.concatenate(recursive_flatten(index), orientation=self.orientation)
            if label is not None:
                label = self.concatenate(recursive_flatten(label), orientation=self.orientation)

        if self.quick_getitem:
            return BeamData.data_batch(data=data, index=index, label=label)

        return self.clone(data=data, index=index, label=label)

    @staticmethod
    def recursive_filter(x, info):

        if info is None:
            return None

        inf_g = info.groupby('fold')
        folds = info['fold'].unique()

        def _recursive_filter(xi, flat_key=0):
            x_type = check_type(xi)
            if x_type.major == 'container':

                keys = []
                values = []
                index = []
                label = []

                for k, v in iter_container(xi):
                    i, v, l, flat_key = _recursive_filter(v, flat_key=flat_key)
                    if len(v):
                        values.append(v)
                        index.append(i)
                        keys.append(k)
                        label.append(l)

                if not is_arange(keys):
                    values = dict(zip(keys, values))
                    index = dict(zip(keys, index))
                    label = dict(zip(keys, label))
                else:
                    values = [values[j] for j in np.argsort(keys)]
                    index = [index[j] for j in np.argsort(keys)]
                    label = [label[j] for j in np.argsort(keys)]

                return index, values, label, flat_key

            else:

                if flat_key not in folds:
                    return [], [], [], flat_key + 1

                info_in_fold = inf_g.get_group(flat_key)

                in_fold_index = info_in_fold['fold_index']
                x_type = check_type(xi)

                label = info_in_fold['label'] if 'label' in info_in_fold else None

                if xi is None:
                    return None, None, None, flat_key + 1
                elif x_type.minor == 'pandas':
                    return in_fold_index.index, xi.iloc[in_fold_index.values], label, flat_key + 1
                elif x_type.minor == 'native':
                    return in_fold_index.index, [xi], label, flat_key + 1
                else:
                    return in_fold_index.index, xi[in_fold_index.values], label, flat_key + 1

        i, d, l, _ = _recursive_filter(x)

        return DataBatch(data=d, index=i, label=l)

    def slice_index(self, index):

        if not self.cached:
            raise LookupError(f"Cannot slice by index as data is not cached")

        if self.orientation in ['simple', 'columns', 'simplified_index']:

            info = None
            if self.label is not None:
                if hasattr(self.label, 'loc'):
                    label = self.label.loc[index]
                else:
                    label = self.label[index]
            else:
                label = None

            if self.orientation in ['simple', 'simplified_index']:
                if hasattr(self.data, 'loc'):
                    data = self.data.loc[index]
                else:
                    data = self.data[index]

            else:

                if self.index is not None:
                    iloc = self.info['map'].loc[index]
                else:
                    iloc = index

                data = recursive_batch(self.data, iloc)

        elif self.orientation in ['index', 'packed']:

            info = self.info.loc[index]
            db = BeamData.recursive_filter(self.data, info)
            data = db.data
            index = db.index
            label = db.label

        else:
            raise ValueError(f"Cannot fetch batch for BeamData with orientation={self.orientation}")

        if self.quick_getitem:
            return BeamData.data_batch(data, index=index, label=label, orientation=self.orientation, info=info)

        return self.clone(data=data, columns=self.columns, index=index, label=label,
                          orientation=self.orientation, info=info)

    @staticmethod
    def data_batch(data, index=None, label=None, orientation=None, info=None):

        data_type = check_type(data, check_element=False, check_minor=False)
        if data_type.major == 'container' and len(data) == 1:
            if isinstance(data, dict):
                key = list(data.keys())[0]
            else:
                key = 0

            data = data[key]
            if index is not None:
                index = index[key]
                if isinstance(label, pd.Series):
                    label = label.values
            if label is not None:
                label = label[key]
                if isinstance(label, pd.Series):
                    label = label.values

        elif data_type.major == 'container':
            data = BeamData._concatenate(data=data, orientation=orientation)

            if index is not None:
                index = BeamData._concatenate(data=index, orientation=orientation)

                if info is not None:
                    flat_index = pd.Series(np.arange(len(info)), index=index)
                    inverse_map = flat_index.loc[info.index].values

            if label is not None:
                label = BeamData._concatenate(data=label, orientation=orientation)

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

        if keys_type.major == 'scalar':
            if replace_missing:
                if keys not in data:
                    return None
            return data[keys]
        elif keys_type.minor == 'tuple':
            if replace_missing:
                return get_closest_item_with_tuple_key(data, keys)
            return get_item_with_tuple_key(data, keys)
        else:
            sliced = [] if data_type.minor == 'list' else {}
            for k in keys:
                if replace_missing:
                    if keys not in data:
                        sliced[k] = None
                        continue
                sliced[k] = data[k]
            return sliced

    def get_simplified_data_by_key(self, key):
        ind = self.info['map'].loc[self.info['fold'] == self.key_map[key]].values
        if hasattr(self.data, 'iloc'):
            data = self.data.iloc[ind]
        else:
            data = self.data[ind]
        label = None
        index = None
        if self.label is not None:
            if hasattr(self.label, 'iloc'):
                label = self.label.iloc[ind]
            else:
                label = self.label[ind]
        if self.index is not None:
            if hasattr(self.index, 'iloc'):
                index = self.index.iloc[ind]
            else:
                index = self.index[ind]

        return DataBatch(data=data, index=index, label=label)

    @staticmethod
    def update_hierarchy(root_path, all_paths):
        @recursive
        def reduce_hierarchy(x):
            if isinstance(x, list):
                return ['/'.join(xi.split('/')[1:]) for xi in x]
            return '/'.join(x.split('/')[1:])

        while True:
            flat_paths = recursive_flatten(all_paths)
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

        if schema_type.major == 'container' and not self.quick_getitem:
            schema = BeamData.slice_scalar_or_list(self.schema, keys, keys_type=keys_type,
                                                   data_type=schema_type, replace_missing=True)
        else:
            schema = self.schema

        if self.stored:

            try:
                all_paths = BeamData.slice_scalar_or_list(self.all_paths, keys, keys_type=keys_type,
                                                          data_type=self.data_type)
                root_path, all_paths = BeamData.update_hierarchy(root_path, all_paths)

            except KeyError:
                raise KeyError(f"Cannot find keys: {keys} in stored BeamData object. "
                               f"If the object is archived you should cache it before slicing.")

        if self.cached:
            if self.orientation == 'simplified_index':
                if keys_type.major == 'scalar':
                    data_batch = self.get_simplified_data_by_key(keys)
                else:
                    data_batch = {k: self.get_simplified_data_by_key(k) for k in keys}
                    data_batch = DataBatch(data={k: v.data for k, v in data_batch.items()},
                                           index={k: v.index for k, v in data_batch.items()},
                                           label={k: v.label for k, v in data_batch.items()})

                if self.quick_getitem and data is not None:
                    return BeamData.data_batch(data_batch.data, index=data_batch.index,
                                               label=data_batch.label, orientation=self.orientation)

                return self.clone(data=data_batch.data, path=root_path, all_paths=all_paths, columns=self.columns,
                                  index=data_batch.index, label=data_batch.label, schema=schema)
            else:
                data = BeamData.slice_scalar_or_list(self.data, keys, keys_type=keys_type, data_type=self.data_type)

        if not self.lazy and self.stored and data is None:

            data = BeamData.read(self.abs_all_paths(all_paths), schema=schema)

        index = self.index
        label = self.label
        # info = self._info

        if self.orientation in ['index', 'packed']:
            if index is not None:
                index = BeamData.slice_scalar_or_list(index, keys, keys_type=keys_type, data_type=self.data_type)
            if label is not None:
                label = BeamData.slice_scalar_or_list(label, keys, keys_type=keys_type, data_type=self.data_type)

        if self.quick_getitem and data is not None:
            return BeamData.data_batch(data, index=index, label=label, orientation=self.orientation)

        # determining orientation and info can be ambiguous so we let BeamData to calculate it
        # from the index and label arguments

        # if self.orientation != 'columns' and info is not None:
        #         info = info.loc[index]
        # return BeamData(data=data, path=all_paths, lazy=self.lazy, columns=self.columns,
        #                 index=index, label=label, orientation=self.orientation, info=info)

        return self.clone(data=data, path=root_path, all_paths=all_paths, columns=self.columns, index=index,
                          label=label, schema=schema)

    def get_default(self, key, default=None):

        if hasattr(self, key) and default is None:
            return getattr(self, key)
        else:
            return default

    def clone(self, *args, data=None, path=None, name=None, all_paths=None, key_map=None,
                 index=None, label=None, columns=None, lazy=None, device=None, target_device=None, schema=None,
                 override=None, compress=None, split_by=None, chunksize=None, chunklen=None, n_chunks=None,
                 partition=None, archive_size=None, preferred_orientation=None, read_kwargs=None, write_kwargs=None,
                 quick_getitem=None, orientation=None, glob_filter=None, info=None, constructor=None, **kwargs):

        name = self.get_default('name', name)
        lazy = self.get_default('lazy', lazy)
        device = self.get_default('device', device)
        target_device = self.get_default('target_device', target_device)
        override = self.get_default('override', override)
        compress = self.get_default('compress', compress)
        split_by = self.get_default('split_by', split_by)
        chunksize = self.get_default('chunksize', chunksize)
        chunklen = self.get_default('chunklen', chunklen)
        n_chunks = self.get_default('n_chunks', n_chunks)
        partition = self.get_default('partition', partition)
        archive_size = self.get_default('archive_size', archive_size)
        preferred_orientation = self.get_default('preferred_orientation', preferred_orientation)
        read_kwargs = self.get_default('read_kwargs', read_kwargs)
        write_kwargs = self.get_default('write_kwargs', write_kwargs)
        quick_getitem = self.get_default('quick_getitem', quick_getitem)
        glob_filter = self.get_default('glob_filter', glob_filter)

        if constructor is None:
            constructor = BeamData

        return constructor(*args, data=data, path=path, name=name, all_paths=all_paths, key_map=key_map,
                 index=index, label=label, columns=columns, lazy=lazy, device=device, target_device=target_device,
                 override=override, compress=compress, split_by=split_by, chunksize=chunksize,
                 chunklen=chunklen, n_chunks=n_chunks, partition=partition, archive_size=archive_size, schema=schema,
                 preferred_orientation=preferred_orientation, read_kwargs=read_kwargs, write_kwargs=write_kwargs,
                 quick_getitem=quick_getitem, orientation=orientation, glob_filter=glob_filter, info=info, **kwargs)

    def inverse_columns_map(self, columns):

        columns_map = self.columns_map
        if check_type(columns).major == 'scalar':
            columns = columns_map[columns]
        else:
            columns = [columns_map[i] for i in columns]

        return columns

    def __repr__(self):
        return self.__str__()

    def __str__(self):

        params = {'orientation': self.orientation, 'lazy': self.lazy, 'stored': self.stored,
                  'cached': self.cached, 'device': self.device, 'objects_type': self.objects_type,
                  'quick_getitem': self.quick_getitem, 'has_index': self.index is not None,
                  'has_label': self.label is not None}
        params_line = ' | '.join([f"{k}: {v}" for k, v in params.items()])

        s = f"BeamData: {self.name}\n"
        s += f"  path: \n"
        s += f"  {self.root_path} \n"
        s += f"  params: \n"
        s += f"  {params_line} \n"
        s += f"  keys: \n"
        s += f"  {self.keys()} \n"
        s += f"  sizes:\n"
        s += f"  {self.size} \n"
        s += f"  shapes:\n"
        s += f"  {self.shape} \n"
        s += f"  types:\n"
        s += f"  {self.dtypes} \n"
        return s

    def __setitem__(self, key, value):
        """
        Set value supports only key hierarchy except for the case of 'simple' orientation.
        @param key:
        @param value:
        """

        key_type = check_type(key)

        if not self.cached or not self.lazy:

            kwargs = self.get_default_params('compress', 'chunksize', 'chunklen', 'n_chunks', 'partition',
                                              'split_by', 'archive_size', 'override')

            path = self.root_path
            all_paths = self.all_paths

            if key_type.major != 'scalar':

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
            path = BeamData.write_object(value, path, **kwargs)
            all_paths[key] = path

        if self.cached:

            if self.orientation in ['simple', 'simplified_index']:
                self.data.__setitem__(key, value)
            else:
                set_item_with_tuple_key(self.data, key, value)

            self._index = None
            self._label = None

            if self.lazy:
                self.stored = False
                self.reset_metadata('all_paths')

    def apply(self, func, *args, **kwargs):
        data = recursive(func)(self.data,  *args, **kwargs)

        return self.clone(data, index=self.index, label=self.label)

    def reset_index(self):
        return self.clone(self.data, index=None, label=self.label, schema=self.schema)

    @staticmethod
    def get_n_chunks(data, n_chunks=None, chunklen=None, chunksize=None, size=None):

        if (n_chunks is None) and (chunklen is None):
            if size is None:
                size = sum(recursive_flatten(recursive_size(data), flat_array=True))
            n_chunks = max(int(np.round(size / chunksize)), 1)
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

    def divide_chunks(self, chunksize=None, chunklen=None, n_chunks=None, partition=None, split_by=None):

            split_by = self.get_default('split_by', split_by)
            chunksize = self.get_default('chunksize', chunksize)
            chunklen = self.get_default('chunklen', chunklen)
            n_chunks = self.get_default('n_chunks', n_chunks)
            partition = self.get_default('partition', partition)

            if not self.cached and split_by != 'keys':

                if not self.lazy:
                    self.cache()
                else:
                    raise ValueError(f"split_by={split_by} is not supported for not-cached and lazay data.")

            if split_by == 'keys':

                if self.cached:

                    for i, (k, d) in enumerate(self.flatten_items.items()):

                        s = BeamData.get_schema_from_tupled_key(self.schema, k)
                        index = None
                        if self.index is not None:
                            index = get_item_with_tuple_key(self.index, k)
                        label = None
                        if self.label is not None:
                            label = get_item_with_tuple_key(self.label, k)

                        info = None
                        if self.info is not None:
                            info = self.info[self.info['fold_index'] == i]
                        yield k, self.clone(d, index=index, label=label, schema=s, info=info)

                else:

                    for i, (k, p) in enumerate(recursive_flatten_with_keys(self.all_paths)):
                        s = get_item_with_tuple_key(self.schema, k)

                        info = None
                        if self.info is not None:
                            info = self.info[self.info['fold_index'] == i]

                        yield k, self.clone(path=self.root_path, all_paths={'data': p}, schema=s, info=info)

            else:

                n_chunks = BeamData.get_n_chunks(self.data, n_chunks=n_chunks, chunklen=chunklen, chunksize=chunksize,
                                                 size=self.total_size)

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

        if self.orientation == 'simple':
            axes = ['index', 'columns', 'else']
        else:
            axes = ['keys', 'index', 'columns', 'else']

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
            if axes[0] == 'keys' and (i_type.minor in ['pandas', 'numpy', 'slice', 'tensor']):
                axes.pop(0)
            if axes[0] == 'keys' and (i_type.minor == 'list' and i_type.element == 'int'):
                axes.pop(0)
            # for orientation == 'simple' we skip the first axis if we slice over columns
            if self.orientation == 'simple' and axes[0] == 'index' and i_type.element == 'str':
                axes.pop(0)

            a = axes.pop(0)
            if a == 'keys':
                obj = obj.slice_keys(ind_i)

            elif a == 'index':

                if i_type.major == 'slice':
                    ind_i = slice_to_index(ind_i, l=len(obj))

                if i_type.element == 'bool':
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
