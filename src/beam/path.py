from pathlib import PurePath, Path
import pandas as pd
import numpy as np
import torch
import scipy
import os
import fastavro
import pyarrow as pa
from argparse import Namespace
import re
from hdfs import InsecureClient
from hdfs.ext.avro import AvroWriter, AvroReader
from hdfs.ext.dataframe import read_dataframe, write_dataframe
import boto3


def beam_path(path, protocol=None, username=None, hostname=None,
              port=None, access_key=None, secret_key=None):
    """

    @param secret_key: AWS secret key
    @param access_key: AWS access key
    @param port:
    @param hostname:
    @param username:
    @param protocol:
    @param path: URI syntax: [protocol://][username@][hostname][:port][/path/to/file]
    @return: BeamPath object
    """

    pattern = re.compile(
        r'^((?P<protocol>[\w]+)://)?((?P<username>[\w]+)@)?(?P<hostname>[\.\w]+)?(:(?P<port>\d+))?(?P<path>/.*)$')
    match = pattern.match(path)
    if match:
        protocol = match.group('protocol')
        username = match.group('username')
        hostname = match.group('hostname')
        port = match.group('port')
        path = match.group('path')

    if protocol is None:
        protocol = 'file'
    if username is None:
        username = ''
    if hostname is None:
        hostname = 'localhost'

    if protocol == 'file':
        return BeamPath(path)

    elif protocol == 's3':

        client = boto3.resource(endpoint_url=f'http://{hostname}:{port}',
                                config=boto3.session.Config(signature_version='s3v4'),
                                verify=False, service_name='s3', aws_access_key_id=access_key,
                                aws_secret_access_key=secret_key)

        return S3Path(path, client=client)

    elif protocol == 'hdfs':
        client = InsecureClient(f'http://{hostname}:{port}', user=username)
        return HDFSPath(path, client=client)

    elif protocol == 'gs':
        raise NotImplementedError
    elif protocol == 'http':
        raise NotImplementedError
    elif protocol == 'https':
        raise NotImplementedError
    elif protocol == 'ftp':
        raise NotImplementedError
    elif protocol == 'ftps':
        raise NotImplementedError
    elif protocol == 'sftp':
        raise NotImplementedError
    else:
        raise NotImplementedError


class PureBeamPath:

    feather_index_mark = "feather_index:"

    def __init__(self, *pathsegments, configuration=None, info=None, **kwargs):
        super().__init__()
        self.path = PurePath(*pathsegments)
        self.configuration = Namespace(**kwargs)
        if configuration is not None:
            for k, v in configuration.items():
                self.configuration[k] = v
        self.info = info if info is not None else {}

    def __str__(self):
        return str(self.path)

    def __repr__(self):
        return self.path.as_uri()

    def parts(self):
        return self.path.parts

    @property
    def name(self):
        return self.path.name

    @property
    def parent(self):
        return self.path.parent

    @property
    def stem(self):
        return self.path.stem

    @property
    def suffix(self):
        return self.path.suffix

    def samefile(self, other):
        raise NotImplementedError

    def is_file(self):
        raise NotImplementedError

    def is_dir(self):
        raise NotImplementedError

    def joinpath(self, *other):
        return self.path.joinpath(*other)

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

            #TODO: add json read with fastavro and shcema
            # x = []
            # with open(path, 'r') as fo:
            #     for record in fastavro.json_reader(fo):
            #         x.append(record)

            x = pd.read_json(path, **kwargs)

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


class S3Path(PureBeamPath):

    def __init__(self, *pathsegments, client=None):
        super().__init__(*pathsegments)
        self.bucket = self.parts[0]
        self.key = self.parts[1]
        self.client = client

    def as_uri(self):
        return f"s3://{self.client.bucket}{str(self)}"

    def __repr__(self):
        return self.as_uri()

    # def exists(self):
    #     return self.s3.exists(str(self))
    #
    # def rename(self, target):
    #     self.s3.rename(str(self), str(target))

    def is_file(self):
        try:
            self.client.Object(self.parts[0], self.parts[1]).load()
            return True
        except self.client.meta.client.exceptions.NoSuchKey:
            return False

    def is_dir(self):
        return not self.is_file()

    def open(self, mode="r", **kwargs):
        if "w" in mode:
            raise NotImplementedError("Writing to S3 is not supported")
        obj = self.client.Object(self.parts[0], self.parts[1]).get()
        return obj["Body"]

    def read_text(self, encoding=None, errors=None):
        obj = self.client.Object(self.parts[0], self.parts[1]).get()
        return obj["Body"].read().decode(encoding, errors)

    def read_bytes(self):
        obj = self.client.Object(self.parts[0], self.parts[1]).get()
        return obj["Body"].read()

    def exists(self):
        try:
            self.client.Object(self.parts[0], self.parts[1]).load()
            return True
        except self.client.meta.client.exceptions.NoSuchKey:
            return False

    def rename(self, target):
        self.client.Object(self.parts[0], self.parts[1]).copy_from(
            CopySource={
                "Bucket": self.parts[0],
                "Key": self.parts[1],
            },
            Bucket=target.parts[0],
            Key=target.parts[1],
        )
        self.unlink()

    def replace(self, target):
        self.rename(target)

    def unlink(self):
        self.client.Object(self.parts[0], self.parts[1]).delete()

    def mkdir(self, parents=False, exist_ok=False):
        if exist_ok and self.exists():
            return
        if self.exists():
            raise FileExistsError("File already exists: %s" % self)
        if not parents:
            raise NotImplementedError("Creating parent directories is not supported")
        self.client.Bucket(self.parts[0]).put_object(Key=self.parts[1] + "/")

    def rmdir(self):
        if self.is_file():
            raise NotADirectoryError("Not a directory: %s" % self)
        self.client.Bucket(self.parts[0]).delete_objects(
            Delete={
                "Objects": [
                    {"Key": key} for key in self.iterdir()
                ]
            }
        )

    def joinpath(self, *args):
        return S3Path(str(super(S3Path, self).joinpath(*args)), client=self.client)

    def iterdir(self):
        bucket = self.client.Bucket(self.parts[0])
        prefix = self.parts[1]
        for obj in bucket.objects.filter(Prefix=prefix):
            yield S3Path("/".join([obj.bucket_name, obj.key]), client=self.client)

    @property
    def parent(self):
        return S3Path(str(super(S3Path, self).parent), client=self.client)

    def write(self, x, **kwargs):

        ext = self.suffix
        # path = str(self)

        if ext == '.fea':
            raise NotImplementedError
        elif ext == '.csv':
            self.client.Object(self.parts[0], self.parts[1]).put(Body=x.to_csv(index=False))
        else:
            raise ValueError("Unsupported extension type.")

    def read(self):

        ext = self.suffix
        path = str(self)

        if ext == '.fea':
            raise NotImplementedError
        elif ext == '.csv':
            obj = self.client.Object(self.parts[0], self.parts[1]).get()
            x = pd.read_csv(obj["Body"])
        else:
            raise ValueError("Unsupported extension type.")

        return x


class HDFSPath(PureBeamPath):

    # TODO: use HadoopFileSystem

    def __init__(self, *pathsegments, client=None, skip_trash=False, n_threads=1,  temp_dir=None, chunk_size=65536,
                 progress=None, cleanup=True):
        super().__init__(*pathsegments, skip_trash=skip_trash, n_threads=n_threads,
                                       temp_dir=temp_dir, chunk_size=chunk_size, progress=progress, cleanup=cleanup)
        self.client = client

    def as_uri(self):
        return f"hdfs://{self.client.url}{str(self)}"

    def __repr__(self):
        return self.as_uri()

    def exists(self):
        return self.client.status(str(self), strict=False) is not None

    def rename(self, target):
        self.client.rename(str(self), str(target))

    def replace(self, target):

        self.client.rename(str(self), str(target))
        return HDFSPath(target, client=self.client)

    def unlink(self, missing_ok=False):
        if not missing_ok:
            self.client.delete(str(self), skip_trash=self.configuration['skip_trash'])
        self.client.delete(str(self), skip_trash=self.configuration['skip_trash'])

    def mkdir(self, mode=0o777, parents=False, exist_ok=False):
        if not exist_ok:
            if self.exists():
                raise FileExistsError
        self.client.makedirs(str(self), permission=mode)

    def rmdir(self):
        self.client.delete(str(self), skip_trash=self.configuration['skip_trash'])

    def joinpath(self, *other):
        return HDFSPath(str(super(HDFSPath, self).joinpath(*other)), client=self.client)

    def iterdir(self):
        files = self.client.list(str(self))
        for f in files:
            yield self.joinpath(f)

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

    @property
    def parent(self):
        return HDFSPath(str(super(HDFSPath, self).parent), client=self.client)

    def glob(self, *args, **kwargs):
        raise NotImplementedError

    def read(self, **kwargs):

        ext = self.suffix
        path = str(self)

        if ext == '.avro':

            x = []
            with AvroReader(self.client, path, **kwargs) as reader:
                self.info['schema'] = reader.writer_schema  # The remote file's Avro schema.
                self.info['content'] = reader.content  # Content metadata (e.g. size).
                for record in reader:
                    x.append(record)

            return x

        elif ext == '.pd':
            x = read_dataframe(self.client, path)

        else:
            raise ValueError(f"Extension type: {ext} not supported for HDFSPath.")

        return x

    def write(self, x, **kwargs):

        ext = self.suffix
        path = str(self)

        if ext == '.avro':

            with AvroWriter(self.client, path) as writer:
                for record in x:
                    writer.write(record)

        elif ext == '.pd':
            write_dataframe(self.client, path, x, **kwargs)
        else:
            raise ValueError(f"Extension type: {ext} not supported for HDFSPath.")

        return self
