from contextlib import contextmanager
from pathlib import PurePath, Path
from .core import PureBeamPath, normalize_host
from io import StringIO, BytesIO
import os
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import pandas as pd
import warnings
from uuid import uuid4 as uuid


class BeamPath(PureBeamPath):

    def __init__(self, *pathsegments, **kwargs):
        scheme = 'file'
        if 'scheme' in kwargs and kwargs['scheme'] is not None:
            scheme = kwargs.pop('scheme')

        PureBeamPath.__init__(self, *pathsegments, scheme=scheme, **kwargs)
        self.path = Path(self.path)

    @classmethod
    def cwd(cls):
        return cls(str(Path.cwd()))

    @classmethod
    def home(cls):
        return cls(str(Path.home()))

    def stat(self):  # add follow_symlinks=False for python 3.10
        return self.path.stat()

    def getmtime(self):
        return os.path.getmtime(str(self.path))

    def chmod(self, mode):
        return self.path.chmod(mode)

    def exists(self):
        return self.path.exists()

    def expanduser(self):
        return self.path.expanduser()

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
        for path in self.path.iterdir():
            yield BeamPath(path)

    def lchmod(self, mode):
        return self.path.lchmod(mode)

    def lstat(self):
        return self.path.lstat()

    def mkdir(self, parents=True, exist_ok=True):
        return self.path.mkdir(parents=parents, exist_ok=exist_ok)

    def owner(self):
        return self.path.owner()

    def read_bytes(self):
        return self.path.read_bytes()

    def read_text(self, *args, **kwargs):
        return self.path.read_text(*args, **kwargs)

    def readlink(self):
        return self.path.readlink()

    def rename(self, target):
        path = self.path.rename(str(target))
        return BeamPath(path)

    def replace(self, target):
        path = self.path.replace(str(target))
        return BeamPath(path)

    def resolve(self, strict=False):

        path = self.path.resolve(strict=strict)
        return self.gen(path)

    def rmdir(self):
        self.path.rmdir()

    def samefile(self, other):
        return self.path.samefile(other)

    def symlink_to(self, target, target_is_directory=False):
        self.path.symlink_to(str(target), target_is_directory=target_is_directory)

    def hardlink_to(self, target):
        self.path.link_to(str(target))

    def link_to(self, target):
        self.path.link_to(str(target))

    def touch(self, *args, **kwargs):
        self.path.touch(*args, **kwargs)

    def unlink(self, missing_ok=False):
        self.path.unlink(missing_ok=missing_ok)

    def write_bytes(self, data):
        return self.path.write_bytes(data)

    def write_text(self, data, *args, **kwargs):
        return self.path.write_text(data, *args, **kwargs)

    def __enter__(self):
        self.file_object = open(self.path, self.mode)
        return self.file_object

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file_object.close()

    def write(self, x, ext=None, **kwargs):

        if ext is None:
            ext = self.suffix

        if ext == '.parquet' and kwargs.get('partition_cols', None) is not None:
            x.to_parquet(str(self), **kwargs)

        return super().write(x, ext=ext, **kwargs)

    def read(self, ext=None, **kwargs):

        if ext is None:
            ext = self.suffix

        if ext == '.parquet' and self.is_dir():
            return pd.read_parquet(str(self), **kwargs)

        return super().read(ext=ext, **kwargs)


class SFTPPath(PureBeamPath):

    def __init__(self, *pathsegments, client=None, hostname=None, username=None, private_key=None, password=None,
                 port=None, private_key_pass=None, ciphers=None, log=False, cnopts=None, default_path=None, **kwargs):

        super().__init__(*pathsegments, scheme='sftp', client=client, hostname=hostname, username=username,
                         private_key=private_key, password=password, port=port, private_key_pass=private_key_pass,
                         ciphers=ciphers, log=log, cnopts=cnopts, default_path=default_path, **kwargs)

        if port is None:
            port = 22
        elif isinstance(port, str):
            port = int(port)

        if client is None:
            import pysftp
            self.client = pysftp.Connection(host=hostname, username=username, private_key=private_key, password=password,
                                            port=port, private_key_pass=private_key_pass, ciphers=ciphers, log=log,
                                            cnopts=cnopts, default_path=default_path)
        else:
            self.client = client

    def samefile(self, other):
        raise NotImplementedError

    def iterdir(self):

        for p in self.client.listdir(remotepath=str(self.path)):
            path = self.path.joinpath(p)
            yield self.gen(path)

    def is_file(self):
        return self.client.isfile(remotepath=str(self.path))

    def is_dir(self):
        return self.client.isdir(remotepath=str(self.path))

    def mkdir(self, *args, mode=777, **kwargs):
        self.client.makedirs(str(self.path), mode=mode)

    def exists(self):
        return self.client.exists(str(self.path))

    def rename(self, target):
        self.client.rename(str(self.path), str(target))

    def __enter__(self):
        if self.mode in ["rb", "r"]:
            self.file_object = self.client.open(str(self.path), self.mode)
        elif self.mode == 'wb':
            self.file_object = BytesIO()
        elif self.mode == 'w':
            self.file_object = StringIO()
        else:
            raise ValueError
        return self.file_object

    def __exit__(self, exc_type, exc_val, exc_tb):

        if self.mode in ["rb", "r"]:
            self.file_object.close()
        else:
            self.file_object.seek(0)
            self.client.putfo(self.file_object, remotepath=str(self.path))
            self.file_object.close()

    def rmdir(self):
        self.client.rmdir(str(self.path))

    def unlink(self, missing_ok=False):

        if self.is_file():
            self.client.remove(str(self.path))
        else:
            raise FileNotFoundError


class S3Path(PureBeamPath):

    def __init__(self, *pathsegments, client=None, hostname=None, port=None, access_key=None,
                 secret_key=None, tls=True, **kwargs):
        super().__init__(*pathsegments, scheme='s3', client=client, hostname=hostname, port=port,
                         access_key=access_key, secret_key=secret_key, tls=tls, **kwargs)

        if not self.is_absolute():
            self.path = PurePath('/').joinpath(self.path)

        if len(self.parts) > 1:
            self.bucket_name = self.parts[1]
        else:
            self.bucket_name = None

        if len(self.parts) > 2:
            self.key = '/'.join(self.parts[2:])
        else:
            self.key = None

        if client is None:

            import boto3

            if type(tls) is str:
                tls = tls.lower() == 'true'

            protocol = 'https' if tls else 'http'
            kwargs = {}
            if hostname is not None:
                kwargs['endpoint_url'] = f'{protocol}://{normalize_host(hostname, port)}'

            if hostname is None and 'region_name' not in kwargs:
                warnings.warn("When working with AWS, please define region_name in kwargs to avoid extra cost")

            client = boto3.resource(config=boto3.session.Config(signature_version='s3v4'),
                                    verify=False, service_name='s3', aws_access_key_id=access_key,
                                    aws_secret_access_key=secret_key, **kwargs)

        self.client = client
        self._bucket = None
        self._object = None

    @property
    def bucket(self):

        if self.bucket_name is None:
            self._bucket = None
        elif self._bucket is None:
            self._bucket = self.client.Bucket(self.bucket_name)
        return self._bucket

    @property
    def object(self):
        if self._object is None:
            self._object = self.client.Object(self.bucket_name, self.key)
        return self._object

    def is_file(self):

        if self.bucket_name is None or self.key is None:
            return False

        key = self.key.rstrip('/')
        return S3Path._exists(self.client, self.bucket_name, key)

    @staticmethod
    def _exists(client, bucket_name, key):
        import botocore
        try:
            # client.Object(bucket_name, key).load()
            client.meta.client.head_object(Bucket=bucket_name, Key=key)
            return True
        except botocore.exceptions.ClientError:
            return False

    def is_dir(self):

        if self.bucket_name is None:
            return True

        if self.key is None:
            return self._check_if_bucket_exists()

        key = self.normalize_directory_key()
        return S3Path._exists(self.client, self.bucket_name, key) or \
               (self._check_if_bucket_exists() and (not self._is_empty(key)))

    def read_text(self, encoding=None, errors=None):
        return self.object.get()["Body"].read().decode(encoding, errors)

    def read_bytes(self):
        return self.object.get()["Body"].read()

    def exists(self):

        if self.key is None:
            return self._check_if_bucket_exists()
        return S3Path._exists(self.client, self.bucket_name, self.key) or self.is_dir()

    def rename(self, target):
        self.object.copy_from(
            CopySource={
                "Bucket": self.bucket_name,
                "Key": self.key,
            },
            Bucket=target.bucket_name,
            Key=target.key,
        )
        self.unlink()

    def _check_if_bucket_exists(self):
        try:
            self.client.meta.client.head_bucket(Bucket=self.bucket_name)
            return True
        except self.client.meta.client.exceptions.ClientError:
            return False

    def replace(self, target):
        self.rename(target)

    def unlink(self, **kwargs):
        if self.is_file():
            self.object.delete()
        if self.is_dir():
            obj = self.client.Object(self.bucket_name, f"{self.key}/")
            obj.delete()

    def mkdir(self, parents=True, exist_ok=True):

        if not parents:
            raise NotImplementedError("parents=False is not supported")

        if exist_ok and self.exists():
            return

        if not self._check_if_bucket_exists():
            self.bucket.create()

        key = self.normalize_directory_key()
        self.bucket.put_object(Key=key)

    def _is_empty_bucket(self):
        for _ in self.bucket.objects.all():
            return False
        return True

    def _is_empty(self, key=None):
        if key is None:
            key = self.key
        for obj in self.bucket.objects.filter(Prefix=key):
            if obj.key.rstrip('/') != self.key.rstrip('/'):
                return False
        return True

    def rmdir(self):

        if self.key is None:
            if not self._is_empty_bucket():
                raise OSError("Directory not empty: %s" % self)
            self.bucket.delete()

        else:
            if self.is_file():
                raise NotADirectoryError("Not a directory: %s" % self)

            if not self._is_empty():
                raise OSError("Directory not empty: %s" % self)

            self.unlink()
            # self.bucket.delete_objects(Delete={"Objects": [{"Key": path.key} for path in self.iterdir()]})

    def key_depth(self, key=None):
        if key is None:
            key = self.key
        if key is None:
            return 0
        return len(list(filter(lambda x: len(x), key.split('/'))))

    def normalize_directory_key(self, key=None):
        if key is None:
            key = self.key
        if key is None:
            return None
        if not key.endswith('/'):
            key += '/'
        return key

    def iterdir(self):

        if self.bucket is None:
            for bucket in self.client.buckets.all():
                yield self.gen(bucket.name)
            return

        key = self.normalize_directory_key()
        if key is None:
            key = ''

        # objects = self.client.meta.client.list_objects_v2(Bucket=self.bucket_name, Prefix=key, Delimiter='/')

        paginator = self.client.meta.client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=self.bucket_name, Prefix=key, Delimiter='/')

        for objects in page_iterator:
            if 'CommonPrefixes' in objects:
                for prefix in objects['CommonPrefixes']:
                    path = f"{self.bucket_name}/{prefix['Prefix']}"
                    yield self.gen(path)

            if 'Contents' in objects:
                for content in objects['Contents']:
                    if content['Key'] == key:
                        continue
                    path = f"{self.bucket_name}/{content['Key']}"
                    yield self.gen(path)

    def __enter__(self):
        if self.mode in ["rb", "r"]:
            file_object = self.client.meta.client.get_object(Bucket=self.bucket_name, Key=self.key)['Body']
            # io_obj = StringIO if 'r' else BytesIO
            self.file_object = BytesIO(file_object.read())
        elif self.mode == 'wb':
            self.file_object = BytesIO()
        elif self.mode == 'w':
            self.file_object = StringIO()
        else:
            raise ValueError

        return self.file_object

    def __exit__(self, exc_type, exc_val, exc_tb):

        if self.mode in ["rb", "r"]:
            self.file_object.close()
        else:
            self.file_object.seek(0)
            self.client.Object(self.bucket_name, self.key).put(Body=self.file_object.getvalue())
            self.file_object.close()


class PyArrowPath(PureBeamPath):

    def __init__(self, *args, strip_path=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.strip_path = strip_path

    @property
    def str_path(self):
        if self.strip_path:
            return str(self).lstrip('/')
        return str(self)

    @property
    def file_info(self):
        return self.client.get_file_info([self.str_path])[0]

    def _exists(self, dir=False, file=False):
        from pyarrow.lib import ArrowIOError
        from pyarrow import fs
        try:
            fi = self.file_info
            if dir:
                return fi.type == fs.FileType.Directory
            if file:
                return fi.type == fs.FileType.File
            return True
        except ArrowIOError:
            return False

    def is_file(self):
        return self._exists(file=True)

    def is_dir(self):
        return self._exists(dir=True)

    def exists(self):
        return self._exists()

    def rename(self, target):
        self.client.move(self.str_path, self.str_path)

    def replace(self, target):
        self.rename(target)

    def unlink(self, **kwargs):
        self.client.delete_file(self.str_path)

    def rmtree(self):
        if self.is_file():
            self.unlink()
        else:
            self.client.delete_dir_contents(self.str_path)

    def mkdir(self, parents=True, exist_ok=True):
        if parents:
            self.client.create_dir(self.str_path, recursive=True)
        else:
            self.client.create_dir(self.str_path, recursive=False)

    def rmdir(self):
        self.client.delete_dir(self.str_path)

    def iterdir(self):

        from pyarrow import fs
        fi = self.client.get_file_info(fs.FileSelector(self.str_path, recursive=False))
        for f in fi:
            yield self.gen(f.path)

    def __enter__(self):
        if self.mode in ["rb", "r"]:
            with self.client.open_input_file(self.str_path) as f:
                content = f.read()
            # io_obj = StringIO if 'r' else BytesIO
            self.file_object = BytesIO(content) if 'b' in self.mode else StringIO(content.decode())
        elif self.mode in ['wb', 'w']:
            self.file_object = BytesIO() if 'b' in self.mode else StringIO()
        else:
            raise ValueError("Invalid mode")

        return self.file_object

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.mode in ["rb", "r"]:
            self.file_object.close()
        else:
            self.file_object.seek(0)
            content = self.file_object.getvalue()
            with self.client.open_output_stream(self.str_path) as f:
                f.write(content if 'b' in self.mode else content.encode())
            self.file_object.close()

    def write(self, x, ext=None, **kwargs):

        if ext is None:
            ext = self.suffix

        if ext == '.parquet':
            import pyarrow.parquet as pq
            pq.write_table(x, self.str_path, filesystem=self.client)

        elif ext == '.orc':
            import pyarrow.orc as orc
            orc.write_table(x, self.str_path, filesystem=self.client)

        return super().write(x, ext=ext, **kwargs)

    def read(self, ext=None, **kwargs):

        if ext is None:
            ext = self.suffix

        if ext == '.parquet':
            import pyarrow.parquet as pq
            return pq.read_table(self.str_path, filesystem=self.client)

        if ext == '.orc':
            import pyarrow.orc as orc
            return orc.read_table(self.str_path, filesystem=self.client)

        return super().read(ext=ext, **kwargs)


class HDFSPath(PureBeamPath):

    def __init__(self, *pathsegments, client=None, hostname=None, port=None, timeout=None,
                 username=None, skip_trash=False, n_threads=0,  temp_dir=None, chunk_size=65536,
                 progress=None, cleanup=True, tls=True, **kwargs):
        super().__init__(*pathsegments, scheme='hdfs', hostname=hostname, port=port, skip_trash=skip_trash,
                                        username=username, n_threads=n_threads, temp_dir=temp_dir, timeout=timeout,
                                        chunk_size=chunk_size, progress=progress, cleanup=cleanup, **kwargs)

        from hdfs import InsecureClient

        if type(tls) is str:
            tls = tls.lower() == 'true'

        protocol = 'https' if tls else 'http'

        if client is None:
            client = InsecureClient(f'{protocol}://{normalize_host(hostname, port)}', user=username)

        self.client = client

    def exists(self):
        return self.client.status(str(self), strict=False) is not None

    def rename(self, target):
        self.client.rename(str(self), str(target))

    def replace(self, target):

        self.client.rename(str(self), str(target))
        return HDFSPath(target, client=self.client)

    def unlink(self, missing_ok=False):
        if not missing_ok:
            self.client.delete(str(self), skip_trash=self['skip_trash'])
        self.client.delete(str(self), skip_trash=self['skip_trash'])

    def mkdir(self, mode=0o777, parents=True, exist_ok=True):
        if not exist_ok:
            if self.exists():
                raise FileExistsError
        if not parents:
            raise NotImplementedError('parents=False not implemented for HDFSPath.mkdir')
        self.client.makedirs(str(self), permission=mode)

    def rmdir(self):
        self.client.delete(str(self), skip_trash=self['skip_trash'])

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

    def read(self, ext=None, **kwargs):

        if ext is None:
            ext = self.suffix

        if ext == '.avro':
            from hdfs.ext.avro import AvroReader
            x = []
            with AvroReader(self.client, str(self), **kwargs) as reader:
                # reader.writer_schema  # The remote file's Avro schema.
                # reader.content  # Content metadata (e.g. size).
                for record in reader:
                    x.append(record)
            return x

        elif ext == '.pd':
            from hdfs.ext.dataframe import read_dataframe
            return read_dataframe(self.client, str(self))

        return super().read(ext=ext, **kwargs)

    def write(self, x, ext=None,  **kwargs):

        if ext is None:
            ext = self.suffix

        if ext == '.avro':
            from hdfs.ext.avro import AvroWriter
            with AvroWriter(self.client, str(self)) as writer:
                for record in x:
                    writer.write(record)

        elif ext == '.pd':
            from hdfs.ext.dataframe import write_dataframe
            write_dataframe(self.client, str(self), x, **kwargs)

        else:
            super().write(x, ext=ext, **kwargs)

    def __enter__(self):
        if self.mode in ["rb", "r"]:

            # chunk_size = self.query['chunk_size']
            # chunk_size = int(chunk_size) if chunk_size is not None else None
            # content = self.client.read(str(self), chunk_size=chunk_size)

            chunk_size = self.query.get('chunk_size', None)
            chunk_size = int(chunk_size) if chunk_size is not None else 0

            with self.client.read(str(self), chunk_size=chunk_size) as reader:
                content = reader.read()

            self.file_object = BytesIO(content) if 'b' in self.mode else StringIO(content.decode())

        elif self.mode in ['wb', 'w']:
            self.file_object = BytesIO() if 'b' in self.mode else StringIO()
        else:
            raise ValueError

        return self.file_object

    def __exit__(self, exc_type, exc_val, exc_tb):

        if self.mode in ["rb", "r"]:
            self.file_object.close()
        else:
            self.file_object.seek(0)
            content = self.file_object.getvalue()
            with self.client.write(str(self)) as writer:
                writer.write(content if 'b' in self.mode else content.encode())
            self.file_object.close()


class S3PAPath(PyArrowPath):
    # a pyarrow implementation of S3Path
    def __init__(self, *pathsegments, client=None, hostname=None, port=None, access_key=None,
                 secret_key=None, tls=True, **kwargs):
        super().__init__(*pathsegments, scheme='s3-pa', client=client, hostname=hostname, port=port,
                         access_key=access_key, secret_key=secret_key, tls=tls, strip_path=True, **kwargs)

        if client is None:

            from pyarrow import fs

            if hostname is not None:
                kwargs['endpoint_override'] = normalize_host(hostname, port)

            if hostname is None and 'region' not in kwargs:
                warnings.warn("When working with AWS, please define region_name in kwargs to avoid extra cost")

            if type(tls) is str:
                tls = tls.lower() == 'true'

            kwargs['scheme'] = 'https' if tls else 'http'

            if 'allow_bucket_creation' not in kwargs:
                kwargs['allow_bucket_creation'] = True
            if 'allow_bucket_deletion' not in kwargs:
                kwargs['allow_bucket_deletion'] = True
            # kwargs['use_virtual_addressing'] = False

            client = fs.S3FileSystem(access_key=access_key, secret_key=secret_key, **kwargs)

        self.client = client


class HDFSPAPath(PyArrowPath):

    # a pyarrow implementation of HDFSPath
    def __init__(self, *pathsegments, client=None, hostname=None, port=None,  username=None, buffer_size=0,
                 replication=3, kerb_ticket=None, extra_conf=None, default_block_size=None, **kwargs):

        super(HDFSPAPath).__init__(*pathsegments, scheme='hdfs-pa', hostname=hostname, port=port,
                         username=username, **kwargs)

        if client is None:
            from pyarrow import fs
            client = fs.HadoopFileSystem(hostname, port=int(port), user=username, replication=replication,
                                         buffer_size=buffer_size, default_block_size=default_block_size,
                                         kerb_ticket=kerb_ticket, extra_conf=extra_conf)

        self.client = client


@contextmanager
def temp_local_file(content, tmp_path='/tmp', name=None, ext=None, binary=True, as_beam_path=True):
    tmp_path = BeamPath(tmp_path).joinpath(uuid())
    tmp_path.mkdir(exist_ok=True, parents=True)
    if name is None:
        name = uuid()
    if ext is not None:
        name = f"{name}{ext}"
    tmp_path = tmp_path.joinpath(name)
    try:
        if binary:
            tmp_path.write_bytes(content)
        else:
            tmp_path.write_text(content)
        yield tmp_path if as_beam_path else str(tmp_path)
    finally:
        tmp_path.unlink()
        tmp_path.parent.rmdir()


class CometAsset(PureBeamPath):
    # a pathlib/beam_path api for comet artifacts

    def __init__(self, *pathsegments, client=None, hostname=None, port=None, access_key=None,
                 secret_key=None, tls=True, **kwargs):
        super().__init__(*pathsegments, scheme='comet', client=client, hostname=hostname, port=port,
                         access_key=access_key, secret_key=secret_key, tls=tls, **kwargs)

        if client is None:

            from comet_ml import API
            client = API(api_key=access_key)

        self.client = client
        parts = self.parts[1:]

        self.workspace = None
        self.project_name = None
        self.experiment_name = None
        self.asset_name = None
        self._experiment_key = None
        self.level = len(parts)
        if self.level > 0:
            self.workspace = parts[0]
        if self.level > 1:
            self.project_name = parts[1]
        if self.level > 2:
            self.experiment_name = parts[2]
        if self.level > 3:
            self.asset_name = parts[3]

    @property
    def experiment_key(self):
        if self.level < 3:
            return None
        if self._experiment_key is None:
            experiments = self.client.get_experiments(self.workspace, self.project_name)
            for e in experiments:
                if e.name == self.experiment_name:
                    self._experiment_key = e.key
                    break
        return self._experiment_key


    @property
    def assets_map(self):
        if self.level < 2:
            return None
        assets_map = self.experiment.get_asset_list()
        return {asset['fileName']: asset for asset in assets_map}

    @property
    def next_level(self):
        kwargs = {}
        args = ()
        attr = 'get'
        if self.level > 0:
            attr = 'get'
            kwargs['workspace'] = self.workspace
        if self.level > 1:
            attr = 'get_experiments'
            kwargs['project_name'] = self.project_name
        if self.level > 2:
            attr = 'get_experiment'
            kwargs = {}
            args = (self.workspace, self.project_name, self.experiment_key,)
        if self.level > 3:
            raise ValueError("CometArtifact: too many levels, it is not a directory")
        return getattr(self.client, attr)(*args, **kwargs)

    def is_file(self):
        assets_map = self.assets_map
        if assets_map is None or self.name not in assets_map:
            return False
        return True

    def is_dir(self):
        return self.level < 4 and self.assets_map is not None

    def exists(self):
        assets_map = self.assets_map
        if assets_map is None:
            return False
        if self.level < 4:
            return True
        return self.name in assets_map

    def iterdir(self):
        if self.level in [0, 1]:
            for p in self.next_level:
                path = self.path.joinpath(p)
                yield self.gen(path)
        elif self.level == 2:
            for e in self.next_level:
                path = self.path.joinpath(e.name)
                yield self.gen(path)
        elif self.level == 3:
            assets_map = self.assets_map
            if assets_map is None:
                return
            for a in assets_map:
                path = self.path.joinpath(a)
                yield self.gen(path)
        else:
            raise ValueError("CometArtifact: too many levels, it is not a directory")

    @property
    def experiment(self):
        if self.level < 3:
            return None
        return self.client.get_experiment(self.workspace, self.project_name, self.experiment_key)

    def mkdir(self, *args, **kwargs):

        if self.level == 1:
            raise ValueError("CometArtifact: cannot create workspace")
        elif self.level == 2:
            self.client.create_project(self.project_name)
        elif self.level == 3:
            from comet_ml import APIExperiment
            exp = APIExperiment(api_key=self.client.api_key, workspace=self.workspace, project_name=self.project_name)
            exp.set_name(self.experiment_name)

        ValueError("CometArtifact: cannot create a directory at this hierarchy level")

    def rmdir(self):
        raise NotImplementedError

    def unlink(self, missing_ok=False):
        if not missing_ok:
            raise NotImplementedError
        if self.level == 1:
            raise ValueError("CometArtifact: cannot delete workspace")
        elif self.level == 2:
            self.client.delete_project(self.project_name)
        elif self.level == 3:
            self.client.delete_experiment(self.next_level.key)
        elif self.level == 4:
            self.next_level.delete_asset(self.assets_map[self.name]['assetId'])
        else:
            raise ValueError("CometArtifact: cannot delete an object at this hierarchy level")

    def rename(self, target):
        raise NotImplementedError

    def replace(self, target):
        raise NotImplementedError

    def __enter__(self):
        if self.mode in ["rb", "r"]:

            content = self.experiment.get_asset(self.assets_map[self.name]['assetId'])
            self.file_object = BytesIO(content) if 'b' in self.mode else StringIO(content.decode())

        elif self.mode in ['wb', 'w']:
            self.file_object = BytesIO() if 'b' in self.mode else StringIO()
        else:
            raise ValueError

        return self.file_object

    def __exit__(self, exc_type, exc_val, exc_tb):

        if self.mode in ["rb", "r"]:
            self.file_object.close()
        else:
            self.file_object.seek(0)
            content = self.file_object.getvalue()
            with temp_local_file(content, name=self.name, as_beam_path=True, binary='b' in self.mode) as tmp_path:
                cwd = os.getcwd()
                try:
                    os.chdir(tmp_path.parent)
                    self.experiment.log_asset(tmp_path.name)
                finally:
                    os.chdir(cwd)
            self.file_object.close()


