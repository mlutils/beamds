from pathlib import PurePosixPath
from urllib.parse import urlparse, urlunparse, parse_qsl, ParseResult
from ..meta import BeamName


class BeamURL(BeamName):

    def __init__(self, url=None, scheme=None, hostname=None, port=None, username=None, password=None, path=None,
                 fragment=None, params=None, **query):

        super().__init__(name=url)

        self._url = url
        self._parsed_url = None
        if url is None:
            netloc = BeamURL.to_netloc(hostname=hostname, port=port, username=username, password=password)
            query = BeamURL.dict_to_query(**query)
            if scheme is None:
                scheme = 'file'
            if path is None:
                path = ''
            if netloc is None:
                netloc = ''
            if query is None:
                query = ''
            if fragment is None:
                fragment = ''
            if params is None:
                params = ''
            self._parsed_url = ParseResult(scheme=scheme, netloc=netloc, path=path, params=params, query=query,
                                           fragment=fragment)

        assert self._url is not None or self._parsed_url is not None, 'Either url or parsed_url must be provided'

    @property
    def parsed_url(self):
        if self._parsed_url is not None:
            return self._parsed_url
        self._parsed_url = urlparse(self._url)
        return self._parsed_url

    @property
    def url(self):
        if self._url is not None:
            return self._url
        self._url = urlunparse(self._parsed_url)
        return self._url

    def __repr__(self):
        return self.url

    def __str__(self):

        netloc = BeamURL.to_netloc(hostname=self.hostname, port=self.port, username=self.username)
        parsed_url = ParseResult(scheme=self.scheme, netloc=netloc, path=self.path, params=None, query=None,
                                 fragment=None)
        return urlunparse(parsed_url)

    def update_query(self, key, value):
        query = self.query
        query[key] = value
        self._parsed_url = self._parsed_url._replace(query=BeamURL.dict_to_query(**query))
        self._url = None

    @property
    def scheme(self):
        return self.parsed_url.scheme

    @property
    def protocol(self):
        return self.scheme

    @property
    def username(self):
        return self.parsed_url.username

    @property
    def hostname(self):
        return self.parsed_url.hostname

    @property
    def password(self):
        return self.parsed_url.password

    @property
    def port(self):
        return self.parsed_url.port

    @property
    def path(self):
        return self.parsed_url.path

    @property
    def query_string(self):
        return self.parsed_url.query

    @property
    def query(self):
        return dict(parse_qsl(self.parsed_url.query))

    @property
    def fragment(self):
        return self.parsed_url.fragment

    @property
    def params(self):
        return self.parsed_url.params

    @staticmethod
    def to_netloc(hostname=None, port=None, username=None, password=None):

        if not hostname:
            return None

        netloc = hostname
        if username:
            if password:
                username = f"{username}:{password}"
            netloc = f"{username}@{netloc}"
        if port:
            netloc = f"{netloc}:{port}"
        return netloc

    @staticmethod
    def to_path(path):
        return PurePosixPath(path).as_posix()

    @staticmethod
    def query_to_dict(query):
        return dict(parse_qsl(query))

    @staticmethod
    def dict_to_query(**query):
        return '&'.join([f'{k}={v}' for k, v in query.items() if v is not None])

    @classmethod
    def from_string(cls, url):
        parsed_url = urlparse(url)
        return cls(url, parsed_url)

    def replace_hostname(self, hostname):
        self._parsed_url = self._parsed_url._replace(netloc=BeamURL.to_netloc(hostname=hostname, port=self.port))
        self._url = None
        return self

    def replace_port(self, port):
        self._parsed_url = self._parsed_url._replace(netloc=BeamURL.to_netloc(hostname=self.hostname, port=port))
        self._url = None
        return self

    def replace_username(self, username):
        self._parsed_url = self._parsed_url._replace(username=username)
        self._url = None
        return self

    def replace_password(self, password):
        self._parsed_url = self._parsed_url._replace(password=password)
        self._url = None
        return self

    def replace_path(self, path):
        self._parsed_url = self._parsed_url._replace(path=path)
        self._url = None
        return self

    def replace_query(self, query):
        self._parsed_url = self._parsed_url._replace(query=query)
        self._url = None
        return self

    def replace_fragment(self, fragment):
        self._parsed_url = self._parsed_url._replace(fragment=fragment)
        self._url = None
        return self

    def replace_params(self, params):
        self._parsed_url = self._parsed_url._replace(params=params)
        self._url = None
        return self

    def replace_scheme(self, scheme):
        self._parsed_url = self._parsed_url._replace(scheme=scheme)
        self._url = None
        return self

    def remove_query(self, key):
        query = self.query
        query.pop(key, None)
        self._parsed_url = self._parsed_url._replace(query=BeamURL.dict_to_query(**query))
        self._url = None
        return self

