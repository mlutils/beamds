from pathlib import PurePath, Path
import botocore
import re
from .utils import PureBeamPath
from io import StringIO, BytesIO


def beam_path(path, protocol=None, username=None, hostname=None, port=None, private_key=None, **kwargs):
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
    if type(path) != str:
        return path

    if ':' not in path:
        return BeamPath(path)

    pattern = re.compile(
        r'^((?P<protocol>[\w]+)://)?((?P<username>[\w\-_]+)@)?(?P<hostname>[\-\.\w]+)?(:(?P<port>\d+))?(?P<path>.*)$')
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

        if private_key is None:
            private_key = Path.home().joinpath('.ssh', 'id_rsa')

        return SFTPPath(path, hostname=hostname, username=username, port=port, private_key=private_key, **kwargs)
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
        for path in self.path.glob(*args, **kwargs):
            yield BeamPath(path)

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
        path = self.path.rename(str(target))
        return BeamPath(path)

    def replace(self, target):
        path = self.path.replace(str(target))
        return BeamPath(path)

    def absolute(self):

        path = self.path.absolute()
        return BeamPath(path)

    def resolve(self, strict=False):

        path = self.path.resolve(strict=strict)
        return BeamPath(path)

    def rglob(self, pattern):
        return self.path.rglob(pattern)

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


def normalize_host(hostname, port=None):

    if hostname is None:
        hostname = 'localhost'
    if port is None:
        host = f"{hostname}"
    else:
        host = f"{hostname}:{port}"

    return host


class SFTPPath(PureBeamPath):

    def __init__(self, *pathsegments, client=None, hostname=None, username=None, private_key=None, password=None,
                 port=None, private_key_pass=None, ciphers=None, log=False, cnopts=None, default_path=None, **kwargs):

        if port is None:
            port = 22

        super().__init__(*pathsegments, client=client, hostname=hostname, username=username, private_key=private_key,
                         password=password, port=port, private_key_pass=private_key_pass, ciphers=ciphers,
                         log=log, cnopts=cnopts, default_path=default_path, **kwargs)

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
            yield SFTPPath(path, client=self.client, configuration=self.configuration)

    def is_file(self):
        return self.client.isfile(remotepath=str(self.path))

    def is_dir(self):
        return self.client.isdir(remotepath=str(self.path))

    def mkdir(self, *args, mode=777, **kwargs):
        self.client.makedirs(str(self.path), mode=mode)

    def exists(self):
        return self.client.exists(str(self.path))

    def glob(self, *args, **kwargs):
        raise NotImplementedError

    def rename(self, target):
        self.client.rename(str(self.path), str(target))

    def __enter__(self):
        if self.mode in ["rb", "r"]:
            self.file_object = self.client.open(str(self.path), self.mode)
        else:
            self.file_object = BytesIO()
        return self.file_object

    def __exit__(self, exc_type, exc_val, exc_tb):

        if self.mode in ["rb", "r"]:
            self.file_object.close()
        else:
            self.client.putfo(self.file_object, remotepath=str(self.path))
            self.file_object.close()

    def rmdir(self):
        self.client.rmdir(str(self.path))

    def unlink(self, missing_ok=False):

        if self.is_file():
            self.client.remove(str(self.path))
        else:
            raise FileNotFoundError

    def as_uri(self):

        return f"sftp://{self.configuration.username}@{self.configuration.hostname}{str(self)}"

    def __repr__(self):
        return self.as_uri()

class S3Path(PureBeamPath):

    def __init__(self, *pathsegments, client=None, hostname=None, port=None, access_key=None,
                 secret_key=None, tls=True, **kwargs):
        super().__init__(*pathsegments, client=client, hostname=hostname, port=port,
                         access_key=access_key, secret_key=secret_key, **kwargs)

        import boto3

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

        if tls:
            self.protocol = 'https'
        else:
            self.protocol = 'http'

        if client is None:
            client = boto3.resource(endpoint_url=f'{self.protocol}://{normalize_host(hostname, port)}',
                                    config=boto3.session.Config(signature_version='s3v4'),
                                    verify=False, service_name='s3', aws_access_key_id=access_key,
                                    aws_secret_access_key=secret_key)

        self.client = client
        self._bucket = None
        self._object = None

    @property
    def bucket(self):
        if self._bucket is None:
            self._bucket = self.client.Bucket(self.bucket_name)
        return self._bucket

    @property
    def object(self):
        if self._object is None:
            self._object = self.client.Object(self.bucket_name, self.key)
        return self._object

    def as_uri(self):

        return f"s3://{self.client.meta.client.meta.endpoint_url}{str(self)}"

    def __repr__(self):
        return self.as_uri()

    def is_file(self):

        key = self.key.rstrip('/')
        return S3Path._exists(self.client, self.bucket_name, key)

    @staticmethod
    def _exists(client, bucket, key):
        try:
            client.Object(bucket, key).load()
            return True
        except botocore.exceptions.ClientError:
            return False

    def is_dir(self):

        if self.key is None:
            return self._check_if_bucket_exists()

        key = self.normalize_directory_key()
        return S3Path._exists(self.client, self.bucket_name, key) or \
               (self._check_if_bucket_exists() and (not self._is_empty(key)))

    def open(self, mode="r", **kwargs):
        if "w" in mode:
            raise NotImplementedError("Writing to S3 is not supported")
        return self.object.get()["Body"]

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

    def unlink(self):
        if self.is_file():
            self.object.delete()
        if self.is_dir():
            obj = self.client.Object(self.bucket_name, f"{self.key}/")
            obj.delete()

    def mkdir(self, parents=True, exist_ok=False):

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

        key = self.normalize_directory_key()

        objects = self.client.meta.client.list_objects_v2(Bucket=self.bucket_name, Prefix=key, Delimiter='/')

        if 'CommonPrefixes' in objects:
            for prefix in objects['CommonPrefixes']:
                yield S3Path(f"{self.bucket_name}/{prefix['Prefix']}", client=self.client,
                             configuration=self.configuration, info=self.info)

        if 'Contents' in objects:
            for content in objects['Contents']:
                yield S3Path(f"{self.bucket_name}/{content['Key']}", client=self.client,
                             configuration=self.configuration, info=self.info)

    # def iterdir(self):
    #     bucket = self.client.Bucket(self.bucket_name)
    #
    #     if self.key is None:
    #         objects = bucket.objects.all()
    #     else:
    #         objects = bucket.objects.filter(Prefix=self.key)
    #
    #     key_depth = self.key_depth()
    #     paths = set()
    #     for obj in objects:
    #
    #         key = list(filter(lambda x: len(x), obj.key.split('/')))
    #         if len(key) <= key_depth:
    #             continue
    #         key = '/'.join(key[:key_depth+1])
    #         if key not in paths:
    #             paths.add(key)
    #             yield S3Path("/".join([obj.bucket_name, key]), client=self.client)

    @property
    def parent(self):
        return S3Path(str(super(S3Path, self).parent), client=self.client)

    def __enter__(self):
        if self.mode in ["rb", "r"]:
            # self.file_object = self.client.Object(self.bucket_name, self.key).get()['Body']
            self.file_object = self.client.meta.client.get_object(Bucket=self.bucket_name, Key=self.key)['Body']

        else:
            self.file_object = BytesIO()
        return self.file_object

    def __exit__(self, exc_type, exc_val, exc_tb):

        if self.mode in ["rb", "r"]:
            self.file_object.close()
        else:
            self.client.Object(self.bucket_name, self.key).put(Body=self.file_object.getvalue())
            self.file_object.close()


class HDFSPath(PureBeamPath):

    # TODO: use HadoopFileSystem

    def __init__(self, *pathsegments, client=None, hostname=None, port=None,
                 username=None, skip_trash=False, n_threads=1,  temp_dir=None, chunk_size=65536,
                 progress=None, cleanup=True):
        super().__init__(*pathsegments, skip_trash=skip_trash, n_threads=n_threads,
                                       temp_dir=temp_dir, chunk_size=chunk_size, progress=progress, cleanup=cleanup)

        from hdfs import InsecureClient

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

        from hdfs.ext.avro import AvroReader
        from hdfs.ext.dataframe import read_dataframe

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

        from hdfs.ext.avro import AvroWriter
        from hdfs.ext.dataframe import write_dataframe

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
