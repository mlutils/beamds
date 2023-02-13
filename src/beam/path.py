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
from .utils import PureBeamPath
from io import StringIO


def beam_path(path, protocol=None, username=None, hostname=None, port=None, **kwargs):
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
        r'^((?P<protocol>[\w]+)://)?((?P<username>[\w]+)@)?(?P<hostname>[\.\w]+)?(:(?P<port>\d+))?(?P<path>.*)$')
    match = pattern.match(path)
    if match:
        protocol = match.group('protocol')
        username = match.group('username')
        hostname = match.group('hostname')
        port = match.group('port')
        path = match.group('path')

    if protocol is None or (protocol == 'file'):
        return BeamPath(path)

    if path == '':
        path = '/'

    if protocol == 's3':
        return S3Path(path, hostname=hostname, port=port,  **kwargs)

    elif protocol == 'hdfs':
        return HDFSPath(path, hostname=hostname, port=port, username=username, **kwargs)

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


class BeamPath(PureBeamPath):

    def __init__(self, *pathsegments, configuration=None, info=None, **kwargs):
        PureBeamPath.__init__(self, *pathsegments, configuration=configuration, info=info, **kwargs)

        if len(pathsegments) == 1 and isinstance(pathsegments[0], PureBeamPath):
            pathsegments = pathsegments[0].parts

        self.path = Path(*pathsegments)

    @classmethod
    def cwd(cls):
        return cls(str(Path.cwd()))

    @classmethod
    def home(cls):
        return cls(str(Path.home()))

    def stat(self):  # add follow_symlinks=False for python 3.10
        return self.path.stat()

    def chmod(self, mode):
        return self.path.chmod(mode)

    def exists(self):
        return self.path.exists()

    def expanduser(self):
        return self.path.expanduser()

    def glob(self, *args, **kwargs):
        return self.path.glob(*args, **kwargs)

    def group(self):
        return self.path.group()

    def is_dir(self):
        return self.path.is_dir()

    def is_file(self):
        return self.path.is_file()

    def is_mount(self):
        return self.path.is_mount()

    def is_symlink(self):
        return self.path.is_symlink()

    def is_socket(self):
        return self.path.is_socket()

    def is_fifo(self):
        return self.path.is_fifo()

    def is_block_device(self):
        return self.path.is_block_device()

    def is_char_device(self):
        return self.path.is_char_device()

    def iterdir(self):
        return self.path.iterdir()

    def lchmod(self, mode):
        return self.path.lchmod(mode)

    def lstat(self):
        return self.path.lstat()

    def mkdir(self, *args, **kwargs):
        return self.path.mkdir(*args, **kwargs)

    def open(self, *args, **kwargs):
        return self.path.open(*args, **kwargs)

    def owner(self):
        return self.path.owner()

    def read_bytes(self):
        return self.path.read_bytes()

    def read_text(self, *args, **kwargs):
        return self.path.read_text(*args, **kwargs)

    def readlink(self):
        return self.path.readlink()

    def rename(self, target):
        return self.path.rename(target)

    def replace(self, target):
        return self.path.replace(target)

    def absolute(self):
        return self.path.absolute()

    def resolve(self, strict=False):
        return self.path.resolve(strict=strict)

    def rglob(self, pattern):
        return self.path.rglob(pattern)

    def rmdir(self):
        return self.path.rmdir()

    def samefile(self, other):
        return self.path.samefile(other)

    def symlink_to(self, target, target_is_directory=False):
        return self.path.symlink_to(target, target_is_directory=target_is_directory)

    def hardlink_to(self, target):
        return self.path.link_to(target)

    def link_to(self, target):
        return self.path.link_to(target)

    def touch(self, *args, **kwargs):
        return self.path.touch(*args, **kwargs)

    def unlink(self, missing_ok=False):
        return self.path.unlink(missing_ok=missing_ok)

    def write_bytes(self, data):
        return self.path.write_bytes(data)

    def write_text(self, data, *args, **kwargs):
        return self.path.write_text(data, *args, **kwargs)

    def __enter__(self):
        self.file_object = open(self.path, self.mode)
        return self.file_object

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file_object.close()


def normalize_host(hostname, port=None):

    if hostname is None:
        hostname = 'localhost'
    if port is None:
        host = f"{hostname}"
    else:
        host = f"{hostname}:{port}"

    return host


class S3Path(PureBeamPath):

    def __init__(self, *pathsegments, client=None, hostname=None, port=None, access_key=None, secret_key=None):
        super().__init__(*pathsegments)

        if not self.is_absolute():
            self.path = PurePath('/').joinpath(self.path)

        if len(self.parts) > 1:
            self.bucket = self.parts[1]
        else:
            self.bucket = None

        if len(self.parts) > 2:
            self.key = '/'.join(self.parts[2:])
        else:
            self.key = None

        if client is None:
            client = boto3.resource(endpoint_url=f'http://{normalize_host(hostname, port)}',
                                    config=boto3.session.Config(signature_version='s3v4'),
                                    verify=False, service_name='s3', aws_access_key_id=access_key,
                                    aws_secret_access_key=secret_key)

        self.client = client

    def as_uri(self):

        return f"s3://{self.client.meta.client.meta.endpoint_url}{str(self)}"

    def __repr__(self):
        return self.as_uri()

    def is_file(self):
        try:
            self.client.Object(self.bucket, self.key).load()
            return True
        except self.client.meta.client.exceptions.NoSuchKey:
            return False

    def is_dir(self):
        return not self.is_file()

    def open(self, mode="r", **kwargs):
        if "w" in mode:
            raise NotImplementedError("Writing to S3 is not supported")
        obj = self.client.Object(self.bucket, self.key).get()
        return obj["Body"]

    def read_text(self, encoding=None, errors=None):
        obj = self.client.Object(self.bucket, self.key).get()
        return obj["Body"].read().decode(encoding, errors)

    def read_bytes(self):
        obj = self.client.Object(self.bucket, self.key).get()
        return obj["Body"].read()

    def exists(self):
        try:
            self.client.Object(self.bucket, self.key).load()
            return True
        except self.client.meta.client.exceptions.NoSuchKey:
            return False

    def rename(self, target):
        self.client.Object(self.bucket, self.key).copy_from(
            CopySource={
                "Bucket": self.bucket,
                "Key": self.key,
            },
            Bucket=target.bucket,
            Key=target.key,
        )
        self.unlink()

    def replace(self, target):
        self.rename(target)

    def unlink(self):
        self.client.Object(*self.parts).delete()

    def mkdir(self, parents=False, exist_ok=False):

        if exist_ok and self.exists():
            return

        if self.bucket is None:
            raise ValueError("Cannot create root directory")

        if self.key is None:
            self.client.Bucket(self.bucket).create()

        else:
            if parents:
                for i in range(2, len(self.parts)-1):
                    S3Path(*self.parts[:i], client=self.client).mkdir(exist_ok=True)


        self.client.Bucket(self.bucket).put_object(Key=self.key)

    def rmdir(self):
        if self.is_file():
            raise NotADirectoryError("Not a directory: %s" % self)
        self.client.Bucket(self.bucket).delete_objects(
            Delete={
                "Objects": [
                    {"Key": key} for key in self.iterdir()
                ]
            }
        )

    def joinpath(self, *args):
        return S3Path(str(super(S3Path, self).joinpath(*args)), client=self.client)

    def iterdir(self):
        bucket = self.client.Bucket(self.bucket)
        for obj in bucket.objects.filter(Prefix=self.key):
            yield S3Path("/".join([obj.bucket_name, obj.key]), client=self.client)

    @property
    def parent(self):
        return S3Path(str(super(S3Path, self).parent), client=self.client)

    def __enter__(self):
        if self.mode == "rb":
            self.file_object = self.client.Object(self.bucket, self.key).get()['Body']
        else:
            self.file_object = StringIO()
        return self.file_object

    def __exit__(self, exc_type, exc_val, exc_tb):

        if self.mode == "rb":
            self.file_object.close()
        else:
            self.client.Object(self.bucket, self.key).put(Body=self.file_object.getvalue())
            self.file_object.close()


class HDFSPath(PureBeamPath):

    # TODO: use HadoopFileSystem

    def __init__(self, *pathsegments, client=None, hostname=None, port=None,
                 username=None, skip_trash=False, n_threads=1,  temp_dir=None, chunk_size=65536,
                 progress=None, cleanup=True):
        super().__init__(*pathsegments, skip_trash=skip_trash, n_threads=n_threads,
                                       temp_dir=temp_dir, chunk_size=chunk_size, progress=progress, cleanup=cleanup)

        if client is None:
            client = InsecureClient(f'http://{normalize_host(hostname, port)}', user=username)

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
