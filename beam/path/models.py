from contextlib import contextmanager
from datetime import datetime
from pathlib import PurePath, Path
from io import StringIO, BytesIO, TextIOWrapper
import os
import urllib3
import stat

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import pandas as pd
import warnings
import tempfile

from .core import PureBeamPath, normalize_host


class BeamPath(PureBeamPath):

    def __init__(self, *pathsegments, **kwargs):
        scheme = 'file'
        if 'scheme' in kwargs and kwargs['scheme'] is not None:
            scheme = kwargs.pop('scheme')

        PureBeamPath.__init__(self, *pathsegments, scheme=scheme, **kwargs)
        self.path = Path(self.path)

    def glob(self, pattern, case_sensitive=None):
        for path in self.path.glob(pattern):
            yield self.gen(path)

    @classmethod
    def cwd(cls):
        return cls(str(Path.cwd()))

    @classmethod
    def home(cls):
        return cls(str(Path.home()))

    def absolute(self):
        return self.gen(self.path.absolute())

    def relative_to(self, *other):
        return self.gen(self.path.relative_to(*other))

    def stat(self):  # add follow_symlinks=False for python 3.10
        return self.path.stat()

    def getmtime(self):
        return os.path.getmtime(str(self.path))

    def getctime(self):
        return os.path.getctime(str(self.path))

    def getatime(self):
        return os.path.getatime(str(self.path))

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

    def rmdir(self):
        self.path.rmdir()

    def samefile(self, other):
        return self.path.samefile(other)

    def symlink_to(self, target, target_is_directory=False):
        self.path.symlink_to(str(target), target_is_directory=target_is_directory)

    def hardlink_to(self, target):
        self.path.hardlink_to(str(target))

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
        self.close_at_exit()

    def write(self, *args, ext=None, **kwargs):

        if ext is None:
            ext = self.suffix

        if ext == '.parquet' and kwargs.get('partition_cols', None) is not None:

            assert len(args) == 1, "Only one argument is allowed for parquet writing with partition_cols"
            df = args[0]
            df.to_parquet(str(self), **kwargs)

        return super().write(*args, ext=ext, **kwargs)

    def read(self, ext=None, **kwargs):

        if ext is None:
            ext = self.suffix

        if ext == '.parquet' and self.is_dir():
            return pd.read_parquet(str(self), **kwargs)

        return super().read(ext=ext, **kwargs)


class SMBPath(PureBeamPath):

    def __init__(self, *pathsegments, client=None, hostname=None, username=None, password=None, port=None,
                 connection_timeout=60, **kwargs):

        if port is None:
            port = 445
        super().__init__(*pathsegments, scheme='smb', client=client, hostname=hostname, username=username,
                         password=password, port=port, **kwargs)
        self.connection_timeout = connection_timeout
        if self.client is None:
            import smbclient
            self.client = smbclient
        self.credentials = {'username': self.username, 'password': self.password, 'port': self.port}
        self.register_smb_session()

    @staticmethod
    def _smb_path(server, path):
        path = path.replace('/', '\\')
        return fr"\\{server.upper()}\{path}"

    @property
    def smb_path(self):
        return SMBPath._smb_path(self.hostname, str(self.path))

    def register_smb_session(self):
        self.client.register_session(self.hostname.upper(), connection_timeout=self.connection_timeout,
                                     **self.credentials)

    def exists(self):
        try:
            self.client.stat(self.smb_path)
            return True
        except FileNotFoundError:
            # The path does not exist
            return False
        except Exception as e:
            # Other exceptions might indicate a problem with the network, permissions, etc.
            # Depending on your application, you might want to handle these differently.
            return False

    def is_file(self):
        try:
            file_attributes = self.client.stat(self.smb_path).st_file_attributes
            # Check if the directory attribute is not set
            return not file_attributes & stat.FILE_ATTRIBUTE_DIRECTORY
        except Exception as e:
            # Handle exceptions, possibly returning False or re-raising
            return False

    def is_dir(self):
        try:
            file_attributes = self.client.stat(self.smb_path).st_file_attributes
            # Check if the directory attribute is set
            return bool(file_attributes & stat.FILE_ATTRIBUTE_DIRECTORY)
        except Exception as e:
            # Handle exceptions
            return False

    def mkdir(self, parents=True, exist_ok=True, **kwargs):

        if self.is_root():
            return

        if not exist_ok:
            if self.exists():
                raise FileExistsError(f"File already exists: {self.smb_path}")

        if parents and not self.parent.exists():
            self.parent.mkdir(parents=True, exist_ok=True, **kwargs)

        from smbprotocol.exceptions import SMBOSError
        from smbprotocol.header import NtStatus

        try:
            self.client.mkdir(self.smb_path, **{**self.credentials, **kwargs})
        except SMBOSError as e:
            if e.ntstatus == NtStatus.STATUS_OBJECT_NAME_COLLISION:
                pass
            else:
                raise e

    def rmdir(self):
        self.client.rmdir(self.smb_path, **self.credentials)

    def unlink(self, missing_ok=False):
        self.client.unlink(self.smb_path, **self.credentials)

    def iterdir(self):
        for p in self.client.listdir(self.smb_path, **self.credentials):
            yield self.gen(f"{self.path}/{p}")

    def replace(self, target):
        self.rename(target)

    def rename(self, target):
        self.client.rename(self.smb_path, target.smb_path, **self.credentials)

    def __enter__(self):
        if self.mode in ["rb", "r"]:
            self.file_object = self.client.open_file(self.smb_path, mode=self.mode, **self.credentials,
                                                     encoding=self.open_kwargs['encoding'],
                                                     newline=self.open_kwargs['newline'],
                                                     errors=self.open_kwargs['errors'])
        elif self.mode == 'wb':
            self.file_object = BytesIO()
        elif self.mode == 'w':
            self.file_object = StringIO(newline=self.open_kwargs['newline'])
        else:
            raise ValueError
        return self.file_object

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.mode in ["wb", "w"]:
            self.file_object.seek(0)
            self.client.open_file(self.smb_path, mode=self.mode, **self.credentials).write(self.file_object.getvalue())
        self.close_at_exit()


class SFTPPath(PureBeamPath):

    def __init__(self, *pathsegments, client=None, hostname=None, username=None, private_key=None, password=None,
                 port=None, private_key_pass=None, ciphers=None, log=False, cnopts=None, default_path=None,
                 disable_hostkey=True, **kwargs):

        super().__init__(*pathsegments, scheme='sftp', client=client, hostname=hostname, username=username,
                         private_key=private_key, password=password, port=port, private_key_pass=private_key_pass,
                         ciphers=ciphers, log=log, cnopts=cnopts, default_path=default_path, **kwargs)

        if port is None:
            port = 22
        elif isinstance(port, str):
            port = int(port)

        if client is None:
            import pysftp
            if disable_hostkey:
                cnopts = pysftp.CnOpts()
                cnopts.hostkeys = None  # This disables host key checking
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
        if self.mode == "rb":
            self.file_object = self.client.open(str(self.path), self.mode)
        elif self.mode == "r":
            self.file_object = TextIOWrapper(self.client.open(str(self.path), self.mode),
                                             encoding=self.open_kwargs['encoding'],
                                            newline=self.open_kwargs['newline'])
        elif self.mode == 'wb':
            self.file_object = BytesIO()
        elif self.mode == 'w':
            self.file_object = StringIO(newline=self.open_kwargs['newline'])
        else:
            raise ValueError
        return self.file_object

    def __exit__(self, exc_type, exc_val, exc_tb):

        if self.mode in ["wb", "w"]:
            self.file_object.seek(0)
            self.client.putfo(self.file_object, remotepath=str(self.path))

        self.close_at_exit()

    def rmdir(self):
        self.client.rmdir(str(self.path))

    def unlink(self, missing_ok=False):

        if self.is_file():
            self.client.remove(str(self.path))
        else:
            raise FileNotFoundError


class S3Path(PureBeamPath):

    def __init__(self, *pathsegments, client=None, hostname=None, port=None, access_key=None,
                 secret_key=None, tls=True, storage_class=None, **kwargs):
        super().__init__(*pathsegments, scheme='s3', client=client, hostname=hostname, port=port,
                         access_key=access_key, secret_key=secret_key, tls=tls, storage_class=storage_class, **kwargs)

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
        self.storage_class = storage_class or 'STANDARD'

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

    def stat(self):
        if not self.exists():
            raise FileNotFoundError(f"No such file or directory: '{self}'")

        metadata = self.client.meta.client.head_object(Bucket=self.bucket_name, Key=self.key)

        metadata = {
            'size': metadata['ContentLength'],  # File size in bytes
            'last_modified': metadata['LastModified'].timestamp(),  # Last modified time as a timestamp
            'etag': metadata['ETag'],  # ETag (often used as a unique identifier for the content)
            'content_type': metadata['ContentType'],  # MIME type of the file
            'storage_class': metadata['StorageClass'],  # Storage class
            'owner': metadata.get('Owner', {}).get('DisplayName', None),  # Owner's display name (if available)
            'permissions': os.stat_result(
                (0, 0, 0, 0, 0, 0, metadata['ContentLength'],
                 metadata['LastModified'].timestamp(), metadata['LastModified'].timestamp(),
                 metadata['LastModified'].timestamp()))
            # Placeholder permissions, can be customized
        }

        # replace the storage class defined in the constructor
        storage_class = metadata['StorageClass']
        if storage_class != self.storage_class:
            self.storage_class = storage_class
            self.url.update_query('storage_class', storage_class)

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

        if self.key is not None:
            key = self.normalize_directory_key()
            self.bucket.put_object(Key=key, StorageClass=self.storage_class)

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
            self.file_object = StringIO(newline=self.open_kwargs['newline'])
        else:
            raise ValueError

        return self.file_object

    def __exit__(self, exc_type, exc_val, exc_tb):

        from botocore.exceptions import ClientError
        if self.mode in ["wb", "w"]:
            self.file_object.seek(0)
            try:
                self.client.Object(self.bucket_name, self.key).put(Body=self.file_object.getvalue(),
                                                                   StorageClass=self.storage_class)
            except ClientError as e:
                from ..logging import BeamError
                raise BeamError(f"Error writing to {self.bucket_name}/{self.key}: {e}, "
                                f"consider changing the storage class (currently {self.storage_class})",
                                error=e)

        self.close_at_exit()

    def getmtime(self):
        d = self.object.get()["LastModified"]
        return d.timestamp()


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
            encoding = self.open_kwargs['encoding'] or 'utf-8'
            self.file_object = BytesIO(content) if 'b' in self.mode else StringIO(content.decode(encoding),
                                                                                  newline=self.open_kwargs['newline'])
        elif self.mode in ['wb', 'w']:
            self.file_object = BytesIO() if 'b' in self.mode else StringIO(newline=self.open_kwargs['newline'])
        else:
            raise ValueError("Invalid mode")

        return self.file_object

    def __exit__(self, exc_type, exc_val, exc_tb):

        if self.mode in ["wb", "w"]:
            self.file_object.seek(0)
            content = self.file_object.getvalue()
            with self.client.open_output_stream(self.str_path) as f:
                f.write(content if 'b' in self.mode else content.encode())

        self.close_at_exit()

    def write(self, *args, ext=None, **kwargs):

        x = None
        if len(args) >= 1:
            x = args[0]

        if ext is None:
            ext = self.suffix

        if ext == '.parquet':
            import pyarrow.parquet as pq
            pq.write_table(x, self.str_path, filesystem=self.client)

        elif ext == '.orc':
            import pyarrow.orc as orc
            orc.write_table(x, self.str_path, filesystem=self.client)

        return super().write(*args, ext=ext, **kwargs)

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

    def write(self, *args, ext=None,  **kwargs):

        x = None
        if len(args) >= 1:
            x = args[0]

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
            super().write(*args, ext=ext, **kwargs)

    def __enter__(self):
        if self.mode in ["rb", "r"]:

            # chunk_size = self.query['chunk_size']
            # chunk_size = int(chunk_size) if chunk_size is not None else None
            # content = self.client.read(str(self), chunk_size=chunk_size)

            chunk_size = self.query.get('chunk_size', None)
            chunk_size = int(chunk_size) if chunk_size is not None else 0

            with self.client.read(str(self), chunk_size=chunk_size) as reader:
                content = reader.read()

            encoding = self.open_kwargs['encoding'] or 'utf-8'
            self.file_object = BytesIO(content) if 'b' in self.mode else StringIO(content.decode(encoding),
                                                                                  newline=self.open_kwargs['newline'])

        elif self.mode in ['wb', 'w']:
            self.file_object = BytesIO() if 'b' in self.mode else StringIO(newline=self.open_kwargs['newline'])
        else:
            raise ValueError

        return self.file_object

    def __exit__(self, exc_type, exc_val, exc_tb):

        if self.mode in ["wb", "w"]:

            self.file_object.seek(0)
            content = self.file_object.getvalue()
            with self.client.write(str(self)) as writer:
                writer.write(content if 'b' in self.mode else content.encode())

        self.close_at_exit()


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


class CometAsset(PureBeamPath):
    # a pathlib/beam_path api for comet artifacts

    def __init__(self, *pathsegments, client=None, hostname=None, port=None, access_key=None,
                 secret_key=None, tls=True, **kwargs):
        super().__init__(*pathsegments, scheme='comet', client=client, hostname=hostname, port=port,
                         access_key=access_key, secret_key=secret_key, tls=tls, **kwargs)

        if self.hostname is not None:
            tls = 'https' if tls else 'http'
            os.environ['COMET_URL_OVERRIDE'] = f"{tls}://{normalize_host(self.hostname, self.port)}/clientlib"

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
            self.client.create_project(self.workspace, self.project_name)
        elif self.level == 3:
            from comet_ml import APIExperiment
            exp = APIExperiment(api_key=self.client.api_key, workspace=self.workspace, project_name=self.project_name)
            exp.set_name(self.experiment_name)

        ValueError("CometArtifact: cannot create a directory at this hierarchy level")

    def rmdir(self):
        raise NotImplementedError

    def unlink(self, missing_ok=False):
        if missing_ok:
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
            encoding = self.open_kwargs['encoding'] or 'utf-8'
            self.file_object = BytesIO(content) if 'b' in self.mode else StringIO(content.decode(encoding),
                                                                                  newline=self.open_kwargs['newline'])

        elif self.mode in ['wb', 'w']:
            self.file_object = BytesIO() if 'b' in self.mode else StringIO(newline=self.open_kwargs['newline'])
        else:
            raise ValueError

        return self.file_object

    def __exit__(self, exc_type, exc_val, exc_tb):

        if self.mode in ["wb", "w"]:
            self.file_object.seek(0)
            content = self.file_object.getvalue()
            from .utils import temp_local_file
            with temp_local_file(content, name=self.name, as_beam_path=True, binary='b' in self.mode) as tmp_path:
                cwd = os.getcwd()
                try:
                    os.chdir(tmp_path.parent)
                    self.experiment.log_asset(tmp_path.name)
                finally:
                    os.chdir(cwd)
        self.close_at_exit()


class RedisPath(PureBeamPath):

    def __init__(self, *pathsegments, client=None, hostname=None, port=None, password=None, username=None, data=None,
                 tls=False, db=0, **kwargs):
        super().__init__(*pathsegments, scheme='redis', client=client, data=data, **kwargs)

        if type(tls) is str:
            tls = tls.lower() == 'true'

        if client is None:

            if port is None:
                port = 6379
            if hostname is None:
                hostname = 'localhost'

            import redis
            client = redis.Redis(host=hostname, port=port, db=db, password=password, username=username, ssl=tls)

        self.client = client

    def normalize_directory_key(self, key=None):
        if key is None:
            key = self.key
        if key is None:
            return None
        if not key.endswith('/'):
            key = f'{key}/'
        return key

    def mkdir(self, *args, parents=True, exist_ok=True):

        if self.is_root():
            return

        if not exist_ok:
            if self.exists():
                raise FileExistsError

        if not parents:
            if not self.parent.exists():
                raise FileNotFoundError
        else:
            p = self.parent
            while not p.root:
                if not p.exists():
                    p.mkdir()
                else:
                    break
                p = p.parent

        self.write_to_redis(self.client, self.directory_key, '')

    def rmdir(self):
        if not self.is_dir():
            raise NotADirectoryError
        if not self.is_empty():
            raise OSError("Directory not empty: %s" % self)
        self.client.delete(self.directory_key)

    def unlink(self, missing_ok=False):
        if self.is_dir():
            raise IsADirectoryError
        self.client.delete(self.key)

    def is_file(self):

        if self.key.endswith('/'):
            return False
        return bool(self.client.exists(self.key))

    def is_dir(self):
        return bool(self.client.exists(self.directory_key))

    def exists(self):
        return bool(self.client.exists(self.key) or self.client.exists(self.directory_key))

    def iterdir(self):
        if not self.is_dir():
            raise NotADirectoryError

        prefix = self.directory_key
        for key in self.client.scan_iter(f'{prefix}*'):
            key = key.decode('utf-8')
            if key.count('/') == prefix.count('/') or (key.endswith('/') and key.count('/') == prefix.count('/') + 1):

                if prefix.rstrip('/') == key.rstrip('/'):
                    continue
                yield self.gen(key)

        # # Pattern for files
        # file_pattern = f'{prefix}[^/]+'
        # for key in self.client.scan_iter(file_pattern):
        #     yield self.gen(key.decode('utf-8'))
        #
        # # Pattern for directories
        # dir_pattern = f'{prefix}[^/]+/'
        # for key in self.client.scan_iter(dir_pattern):
        #     yield self.gen(key.decode('utf-8'))

    def is_empty(self):
        for _ in self.iterdir():
            return False
        return True

    def rename(self, target):
        self.client.rename(self.key, target.key)

    def replace(self, target):
        if target.exists():
            target.unlink()
        self.rename(target)

    @property
    def directory_key(self):
        return self.normalize_directory_key(self.key)

    @property
    def key(self):
        # key = '/'.join(self.parts)
        # # to remove the leading slash if exists
        # key = key.lstrip('/')
        key = str(self)
        return key

    @property
    def _obj(self):
        obj = self.client.hget(self.key, 'data')
        return obj

    @property
    def _timestamp(self):
        timestamp = self.client.hget(self.key, 'modified')
        return timestamp

    def __enter__(self):
        if self.mode in ["rb", "r"]:
            content = self._obj
            encoding = self.open_kwargs['encoding'] or 'utf-8'
            self.file_object = BytesIO(content) if 'b' in self.mode else StringIO(content.decode(encoding),
                                                                                  newline=self.open_kwargs['newline'])
        elif self.mode in ['wb', 'w']:
            self.file_object = BytesIO() if 'b' in self.mode else StringIO(newline=self.open_kwargs['newline'])
        else:
            raise ValueError

        return self.file_object

    @staticmethod
    def write_to_redis(client, key, content, timestamp=None):
        if timestamp is None:
            timestamp = str(datetime.now())
        client.hset(key, 'data', content)
        client.hset(key, 'modified', timestamp)

    def __exit__(self, exc_type, exc_val, exc_tb):

        if self.mode in ['wb', 'w']:
            self.file_object.seek(0)
            content = self.file_object.getvalue()
            self.write_to_redis(self.client, self.key, content)

        self.close_at_exit()

    def glob(self, pattern, case_sensitive=None):
        full_pattern = f'{self.directory_key}{pattern}'
        return [self.gen(key) for key in self.client.scan_iter(full_pattern)]


class MLFlowPath(PureBeamPath):

    def __init__(self, *pathsegments, client=None, hostname=None, port=None, access_key=None,
                secret_key=None, tls=False, **kwargs):
        super().__init__(*pathsegments, scheme='mlflow', client=client, hostname=hostname, port=port,
                        access_key=access_key, secret_key=secret_key, tls=tls, **kwargs)

        if type(tls) is str:
            tls = tls.lower() == 'true'

        if client is None:

            # from mlflow.tracking import MlflowClient
            from mlflow.client import MlflowClient
            if hostname is None:
                if 'MLFLOW_TRACKING_URI' in os.environ:
                    hostname = os.environ['MLFLOW_TRACKING_URI']
                else:
                    hostname = 'localhost'
                    if port is None:
                        from ..utils import beam_service_port
                        port = beam_service_port('MLFLOW_PORT')
                    if port is None:
                        port = 80
            tls = 'https' if tls else 'http'

            client = MlflowClient(tracking_uri=f'{tls}://{normalize_host(hostname, port)}')

        self.client = client

        self._experiment = None
        self._run = None
        self._tmp_dir = None

        parts = self.parts[1:]

        self.experiment_name = None
        self.run_name = None
        self.artifact_path = None
        self.artifact_dir = None
        self.artifact_name = None

        self.level = len(parts)
        if self.level > 0:
            self.experiment_name = parts[0]
        if self.level > 1:
            self.run_name = parts[1]
        if self.level > 2:
            self.artifact_path = '/'.join(parts[2:])
            self.artifact_dir = '/'.join(parts[2:-1])
            self.artifact_name = parts[-1]

    @property
    def experiment(self):
        if self._experiment is None:
            if self.experiment_name is not None:
                self._experiment = self.client.get_experiment_by_name(self.experiment_name)
        return self._experiment

    @property
    def experiment_id(self):
        if self.experiment is None:
            return None
        return self.experiment.experiment_id

    @property
    def run_id(self):
        if self.run is None:
            return None
        return self.run.info.run_id

    @property
    def run(self):
        if self._run is None:
            if self.run_name is not None:
                for r in self.client.search_runs([self.experiment_id],
                                                 f"tags.mlflow.runName = '{self.run_name}'"):
                    self._run = r
                    break
        return self._run

    def mkdir(self, *args, parents=True, exist_ok=True):

        assert parents and exist_ok, "Only parents=True and exist_ok=True are supported"
        if self.experiment_name is not None:
            if self.experiment is None:
                exp_id = self.client.create_experiment(self.experiment_name)
                self._experiment = self.client.get_experiment(exp_id)

        if self.run_name is not None:
            if self.run is None:
                self._run = self.client.create_run(self.experiment_id, run_name=self.run_name)

        if self.artifact_path is not None:
            with tempfile.TemporaryDirectory() as tmp_dir:
                p = BeamPath(tmp_dir).joinpath(self.artifact_name)
                p.mkdir()
                self.client.log_artifact(self.run_id, p.str, artifact_path=self.artifact_dir)

    def exists(self):
        if self.level == 0:
            return True
        if self.level == 1:
            return self.experiment is not None
        if self.level == 2:
            return self.run is not None
        if self.level > 2:
            return self.artifact_info is not None

    def __enter__(self):
        if self.mode in ["rb", "r"]:
            self._tmp_dir = tempfile.TemporaryDirectory()
            v = os.environ.get('MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR', 'True')
            os.environ['MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR'] = 'False'
            self.client.download_artifacts(self.run_id, self.artifact_path, self._tmp_dir.name)
            os.environ['MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR'] = v
            self.file_object = os.path.join(self._tmp_dir.name, self.artifact_name)

        elif self.mode in ['wb', 'w']:
            self._tmp_dir = tempfile.TemporaryDirectory()
            self.file_object = os.path.join(self._tmp_dir.name, self.artifact_name)
        else:
            raise ValueError

        return self.file_object

    def __exit__(self, exc_type, exc_val, exc_tb):

        if self.mode in ["wb", "w"]:
            self.client.log_artifact(self.run_id, self.file_object, artifact_path=self.artifact_dir)
            self._tmp_dir.cleanup()
        elif self.mode in ["rb", "r"]:
            self._tmp_dir.cleanup()
        else:
            raise ValueError

    def iterdir(self):
        if self.level == 0:
            for e in self.client.list_experiments():
                yield self.gen(f"/{e.name}")
        if self.level == 1:
            for r in self.client.search_runs([self.experiment_id]):
                yield self.gen(f"/{self.experiment_name}/{r.info.run_name}")
        if self.level >= 2:
            for a in self.client.list_artifacts(self.run_id, self.artifact_path):
                yield self.gen(f"/{self.experiment_name}/{self.run_name}/{a.path}")

    def rmdir(self):
        raise NotImplementedError("MLFlowPath: rmdir is not supported")

    def unlink(self, missing_ok=False):
        raise NotImplementedError("MLFlowPath: unlink is not supported")

    @property
    def artifact_info(self):

        if self.run_id is None:
            return None
        if self.artifact_dir is None:
            return None
        artifacts = self.client.list_artifacts(self.run_id, self.artifact_dir)
        if not artifacts:
            return None
        for a in artifacts:
            if a.path == self.artifact_path:
                return a
        return None

    def is_file(self):

        info = self.artifact_info
        if self.artifact_info is None:
            return False
        return not info.is_dir

    def is_dir(self):

        info = self.artifact_info
        if self.artifact_info is None:
            return False
        return info.is_dir

    def stat(self):
        return {'size': self.artifact_info.file_size,
                'is_dir': self.artifact_info.is_dir,
                }

    def rename(self, target):
        raise NotImplementedError("MLFlowPath: rename is not supported")


class GoogleStoragePath(PureBeamPath):

    def __init__(self, *pathsegments, client=None, hostname=None, port=None, access_key=None,
                 project_id=None, tls=True, **kwargs):
        super().__init__(*pathsegments, scheme='gs', client=client, hostname=hostname, port=port,
                         access_key=access_key, project_id=project_id, tls=tls, **kwargs)

        if not self.is_absolute():
            self.path = PurePath('/').joinpath(self.path)

        # Parse the path to get bucket name and key
        parts = self.path.parts
        if len(parts) > 1:
            self.bucket_name = parts[1]
            if len(parts) > 2:
                self.key = '/'.join(parts[2:])
            else:
                self.key = None
        else:
            self.bucket_name = None
            self.key = None

        if client is None:
            from google.cloud import storage

            if hostname is not None:
                scheme = 'https' if tls else 'http'
                kwargs['endpoint_url'] = f'{scheme}://{normalize_host(hostname, port)}'
            
            if access_key is not None:
                # If service account file is provided, use it
                client = storage.Client.from_service_account_json(
                    credentials_file=access_key,
                    project=project_id  # project_id is optional here
                )
            else:
                # Use default credentials
                client = storage.Client(project=project_id)  # project_id is optional here

        self.client = client
        self._bucket = None
        self._object = None

    @property
    def bucket(self):
        if self.bucket_name is None:
            self._bucket = None
        elif self._bucket is None:
            self._bucket = self.client.bucket(self.bucket_name)
        return self._bucket

    @property
    def object(self):
        if self._object is None and self.key is not None:
            self._object = self.bucket.blob(self.key)
        return self._object

    def exists(self):
        if self.bucket_name is None:
            return True
        if self.key is None:
            return self._check_if_bucket_exists()
        return self._exists(self.bucket_name, self.key) or self.is_dir()

    def _check_if_bucket_exists(self):
        try:
            self.client.get_bucket(self.bucket_name)
            return True
        except Exception:
            return False

    def _exists(self, bucket_name, key):
        try:
            self.client.get_bucket(bucket_name).get_blob(key)
            return True
        except Exception:
            return False

    def is_file(self):
        if self.bucket_name is None or self.key is None:
            return False
        key = self.key.rstrip('/')
        return self._exists(self.bucket_name, key)

    def is_dir(self):
        if self.bucket_name is None:
            return True
        if self.key is None:
            return self._check_if_bucket_exists()
        key = self.normalize_directory_key()
        return self._exists(self.bucket_name, key) or \
               (self._check_if_bucket_exists() and (not self._is_empty(key)))

    def normalize_directory_key(self, key=None):
        if key is None:
            key = self.key
        if key is None:
            return None
        if not key.endswith('/'):
            key += '/'
        return key

    def _is_empty(self, key=None):
        if key is None:
            key = self.key
        for blob in self.bucket.list_blobs(prefix=key):
            if blob.name.rstrip('/') != self.key.rstrip('/'):
                return False
        return True

    def mkdir(self, parents=True, exist_ok=True):
        if not parents:
            raise NotImplementedError("parents=False is not supported")

        if exist_ok and self.exists():
            return

        if not self._check_if_bucket_exists():
            self.bucket.create()

        if self.key is not None:
            key = self.normalize_directory_key()
            self.bucket.blob(key).upload_from_string('')

    def rmdir(self):
        if self.key is None:
            if not self._is_empty():
                raise OSError("Directory not empty: %s" % self)
            self.bucket.delete()
        else:
            if self.is_file():
                raise NotADirectoryError("Not a directory: %s" % self)
            if not self._is_empty():
                raise OSError("Directory not empty: %s" % self)
            self.unlink()

    def unlink(self, missing_ok=False):
        if self.is_file():
            self.object.delete()
        if self.is_dir():
            obj = self.bucket.blob(f"{self.key}/")
            obj.delete()

    def rename(self, target):
        self.object.copy_to(target.object)
        self.unlink()

    def replace(self, target):
        self.rename(target)

    def iterdir(self):
        if self.bucket is None:
            for bucket in self.client.list_buckets():
                yield self.gen(bucket.name)
            return

        key = self.normalize_directory_key()
        if key is None:
            key = ''

        # Explicitly iterate pages to fetch prefixes
        iterator = self.client.list_blobs(
            bucket_or_name=self.bucket_name,
            prefix=key,
            delimiter='/',
            include_trailing_delimiter=True
        )

        for page in iterator.pages:
            # First, yield subdirectories from prefixes explicitly
            for prefix in page.prefixes:
                path = f"{self.bucket_name}/{prefix.rstrip('/')}"
                yield self.gen(path)

            # Then yield blobs (files) directly under this prefix
            for blob in page:
                if blob.name == key or blob.name.endswith('/'):
                    continue  # Skip self and explicit folder placeholders
                path = f"{self.bucket_name}/{blob.name}"
                yield self.gen(path)

    def read_bytes(self):
        return self.object.download_as_bytes()

    def read_text(self, encoding=None, errors=None):
        return self.object.download_as_text(encoding=encoding)

    def write_bytes(self, data):
        self.object.upload_from_string(data)

    def write_text(self, data, encoding=None, errors=None):
        self.object.upload_from_string(data, content_type='text/plain')

    def __enter__(self):
        if self.mode in ["rb", "r"]:
            encoding = self.open_kwargs['encoding'] or 'utf-8'
            content = self.read_bytes() if 'b' in self.mode else self.read_text(encoding=encoding)
            self.file_object = BytesIO(content) if 'b' in self.mode else StringIO(content,
                                                                                  newline=self.open_kwargs['newline'])
        elif self.mode in ['wb', 'w']:
            self.file_object = BytesIO() if 'b' in self.mode else StringIO(newline=self.open_kwargs['newline'])
        else:
            raise ValueError

        return self.file_object

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.mode in ["wb", "w"]:
            self.file_object.seek(0)
            content = self.file_object.getvalue()
            if 'b' in self.mode:
                self.write_bytes(content)
            else:
                self.write_text(content)
        self.close_at_exit()

    def stat(self):
        if not self.exists():
            raise FileNotFoundError(f"No such file or directory: '{self}'")

        self.object.reload()  # Fetch latest metadata from GCS

        return {
            'size': self.object.size,
            'last_modified': self.object.updated.timestamp() if self.object.updated else datetime.now().timestamp(),
            'etag': self.object.etag,
            'content_type': self.object.content_type,
            'owner': self.object.owner.get('entity') if self.object.owner else None,
            'permissions': os.stat_result((
                0, 0, 0, 0, 0, 0,
                self.object.size,
                self.object.updated.timestamp() if self.object.updated else datetime.now().timestamp(),
                self.object.updated.timestamp() if self.object.updated else datetime.now().timestamp(),
                self.object.updated.timestamp() if self.object.updated else datetime.now().timestamp(),
            ))
        }
    def getmtime(self):
        return self.object.updated.timestamp() if self.object else None
