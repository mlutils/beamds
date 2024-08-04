from typing import Union

from ..meta import BeamName
from .beam_url import BeamURL

resource_names = {
    'path': ['file', 's3', 's3-pa', 'hdfs', 'hdfs-pa', 'sftp', 'comet', 'io', 'dict', 'redis', 'smb', 'nt',
             'mlflow'],
    'serve': ['beam-http', 'beam-https', 'beam-grpc', 'beam-grpcs', 'http', 'https', 'grpc', 'grpcs'],
    'distributed': ['async-http', 'async-https'],
    'llm': ['openai', 'vllm', 'tgi', 'fastchat', 'huggingface', 'samurai', 'samur-openai', 'fastapi-dp'],
    'triton': ['triton', 'triton-http', 'triton-grpc', 'triton-https', 'triton-grpcs'],
    'ray': ['ray'],
    'embedding': ['emb-openai', 'emb-stt'],
}


class BeamResource(BeamName):
    """
    Base class for all resources. Gets as an input a URI and the resource type and returns the resource.
    """

    def __init__(self, resource_type: str = None, url: Union[BeamURL, str] = None, scheme: str = None, hostname: str = None,
                 port: int = None, username: str = None, password: str = None, fragment: str = None, params: str = None,
                 path: str = None, **kwargs):

        super().__init__()
        if isinstance(url, str):
            url = BeamURL(url)

        if url is not None:
            scheme = scheme or url.scheme
            hostname = hostname or url.hostname
            port = port or url.port
            username = username or url.username
            password = password or url.password
            fragment = fragment or url.fragment
            params = params or url.params
            kwargs = kwargs or url.query
            path = path or url.path

        kwargs = {k: v for k, v in kwargs.items() if v is not None and not k.startswith('_')}

        self.url = BeamURL(scheme=scheme, hostname=hostname, port=port, username=username, fragment=fragment,
                           params=params, password=password, path=path, **kwargs)

        self.resource_type = resource_type
        self.scheme = self.url.scheme

    def as_uri(self):
        return self.url.url

    @property
    def hostname(self):
        return self.url.hostname

    @property
    def port(self):
        return self.url.port

    @property
    def username(self):
        return self.url.username

    @property
    def password(self):
        return self.url.password

    @property
    def fragment(self):
        return self.url.fragment

    @property
    def params(self):
        return self.url.params

    @property
    def query(self):
        return self.url.query

    @property
    def is_beam_client(self):
        return self.scheme in resource_names['serve']

    @property
    def str(self):
        return str(self.as_uri())

    def __str__(self):
        return self.str
