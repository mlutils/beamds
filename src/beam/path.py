from pathlib import PurePath, Path
import pandas as pd
import numpy as np
import torch
import scipy
import os
import fastavro
import pyarrow as pa
from argparse import Namespace


class PureBeamPath(PurePath):

    feather_index_mark = "feather_index:"

    def __init__(self, *pathsegments, **kwargs):
        super().__init__(*pathsegments)
        self.configuration = Namespace(**kwargs)

    def samefile(self, other):
        raise NotImplementedError

    def is_file(self):
        raise NotImplementedError

    def is_dir(self):
        raise NotImplementedError

    def joinpath(self, *other):
        raise NotImplementedError

    def mkdir(self, *args, **kwargs):
        raise NotImplementedError

    def exists(self):
        raise NotImplementedError

    def glob(self, *args, **kwargs):
        raise NotImplementedError


class BeamPath(Path, PureBeamPath):

    def __init__(self, *pathsegments):
        Path.__init__(self, *pathsegments)
        PureBeamPath.__init__(self, *pathsegments)

    def read(self, **kwargs):

        ext = self.suffix

        path = str(self)

        if ext == '.fea':
            x = pd.read_feather(path, **kwargs)

            c = x.columns
            for ci in c:
                if PureBeamPath.feather_index_mark in ci:
                    index_name = ci.lstrip(PureBeamPath.feather_index_mark)
                    x = x.rename(columns={ci: index_name})
                    x = x.set_index(index_name)
                    break

        elif ext == '.csv':
            x = pd.read_csv(path, **kwargs)
        elif ext in ['.pkl', '.pickle']:
            x = pd.read_pickle(path, **kwargs)
        elif ext in ['.npy', '.npz']:
            x = np.load(path, allow_pickle=True, **kwargs)
        elif ext == '.scipy_npz':
            x = scipy.sparse.load_npz(path, **kwargs)
        elif ext == '.parquet':
            x = pd.read_parquet(path, **kwargs)
        elif ext == '.pt':
            x = torch.load(path, **kwargs)
        elif ext in ['.xls', '.xlsx', '.xlsm', '.xlsb', '.odf', '.ods', '.odt']:
            x = pd.read_excel(path, **kwargs)
        elif ext == '.avro':
            x = []
            with open(path, 'rb') as fo:
                for record in fastavro.reader(fo):
                    x.append(record)
        elif ext == '.json':
            x = []
            with open(path, 'r') as fo:
                for record in fastavro.json_reader(fo):
                    x.append(record)
        elif ext == '.orc':
            x = pa.orc.read(path, **kwargs)

        else:
            raise ValueError("Unknown extension type.")

        return x

    def write(self, x, **kwargs):

        ext = self.suffix
        path = str(self)

        if ext == '.fea':
            x = pd.DataFrame(x)
            index_name = x.index.name if x.index.name is not None else 'index'
            df = x.reset_index()
            new_name = PureBeamPath.feather_index_mark + index_name
            x = df.rename(columns={index_name: new_name})
            x.to_feather(path, **kwargs)
        elif ext == '.csv':
            x = pd.DataFrame(x)
            x.to_csv(path, **kwargs)
        elif ext in ['.pkl', '.pickle']:
            pd.to_pickle(x, path, **kwargs)
        elif ext == '.npy':
            np.save(path, x, **kwargs)
        elif ext == '.npz':
            np.savez(path, x, **kwargs)
        elif ext == '.scipy_npz':
            scipy.sparse.save_npz(path, x, **kwargs)
            os.rename(f'{path}.npz', path)
        elif ext == '.parquet':
            x = pd.DataFrame(x)
            x.to_parquet(path, **kwargs)
        elif ext == '.pt':
            torch.save(x, path, **kwargs)
        else:
            raise ValueError("Unsupported extension type.")

        return self


class HDFSPath(PureBeamPath):

    def __init__(self, *pathsegments, client=None, skip_trash=False, n_threads=1,  temp_dir=None, chunk_size=65536,
                 progress=None, cleanup=True):
        super(HDFSPath, self).__init__(*pathsegments, skip_trash=skip_trash, n_threads=n_threads,
                                       temp_dir=temp_dir, chunk_size=chunk_size, progress=progress, cleanup=cleanup)

        self.client = client if client is not None else hdfs3.HDFileSystem(n_threads=n_threads,
                                                                          temp_dir=temp_dir,
                                                                          chunk_size=chunk_size,
                                                                          progress=progress,
                                                                          cleanup=cleanup)

    def exists(self):
        return self.client.status(str(self), strict=False) is not None

    def rename(self, target):
        self.client.rename(str(self), str(target))

    def replace(self, target):

        self.client.rename(str(self), str(target))
        return HDFSPath(target, client=self.client)

    def unlink(self, missing_ok=False):
        if not missing_ok:
            self.client.delete(str(self), skip_trash=self.skip_trash)
        self.client.delete(str(self), skip_trash=self.skip_trash)

    def mkdir(self, mode=0o777, parents=False, exist_ok=False):
        if not exist_ok:
            if self.exists():
                raise FileExistsError
        self.client.makedirs(str(self), permission=mode)

    def rmdir(self):
        self.client.delete(str(self), skip_trash=self.skip_trash)

    def iterdir(self):
        files = self.client.list(str(self))
        for f in files:
            yield HDFSPath(self.joinpath(f), client=self.client)

    def samefile(self, other):
        raise NotImplementedError

    def is_file(self):

        status = self.client.status(str(self), strict=False)
        if status is None:
            return False
        return status['type'] == 'FILE'

    def is_dir(self):

        status = self.client.status(str(self), strict=False)
        if status is None:
            return False
        return status['type'] == 'DIRECTORY'

    def glob(self, *args, **kwargs):
        raise NotImplementedError
