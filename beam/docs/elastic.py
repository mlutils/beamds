from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from elasticsearch_dsl import Search, Q
from mlflow.store.artifact.artifact_repository_registry import scheme

from ..path import BeamPath, normalize_host
from ..utils import lazy_property as cached_property

from .core import BeamDoc
from .utils import parse_kql_to_dsl


class BeamElastic(BeamPath, BeamDoc):

    def __init__(self, *args, hostname=None, port=None, username=None, password=None, verify=False,
                 tls=False, client=None, **kwargs):
        super().__init__(*args, hostname=hostname, port=port, username=username, password=password,
                         scheme='elastic', **kwargs)
        self.verify = verify
        self.tls = tls
        self.client = client or self._get_client()

    def _get_client(self):
        protocol = 'https' if self.tls else 'http'
        host = f"{protocol}://{normalize_host(self.hostname, self.port)}"
        return Elasticsearch([host], http_auth=(self.username, self.password), verify_certs=self.verify)

    def _search_index(self, index):
        return Search(using=self.client, index=index)

    @property
    def index(self):
        return self.parts[0]

    @property
    def _search(self):
        return self._search_index(self.index)

    @property
    def query(self):

        if self.path_type in ['root', 'index']:
            return BeamElastic.match_all

        qs = self.parts[1:-1]
        q = BeamElastic.match_all

        for part in qs:
            q = q & parse_kql_to_dsl(part)

        if self.path_type == 'document':
            q = q & Q('ids', values=[self.parts[-1]])
        else:
            q = q & parse_kql_to_dsl(self.parts[-1])

        return q

    @property
    def path_type(self):
        if len(self.parts) == 0:
            return 'root'
        elif len(self.parts) == 1:
            return 'index'
        elif self._is_file_path:
            return 'document'
        else:
            return 'query'

    @property
    def _is_file_path(self):
        # check if last part of the path is _id value
        p = self.parts[-1]
        return ':' not in p and '.' not in p

    @staticmethod
    @property
    def match_all():
        return Q('match_all')

    @staticmethod
    @property
    def match_none():
        return Q('match_none')

    def index_exists(self, index):
        return self.client.indices.exists(index=index)

    def create_index(self, index, body):
        return self.client.indices.create(index=index, body=body)

    def delete_index(self, index):
        return self.client.indices.delete(index=index)

    def index_document(self, index, body, id=None):
        return self.client.index(index=index, body=body, id=id)

    def bulk_index(self, index, docs):
        return bulk(self.client, docs, index=index)

    def search(self, index, body):
        return self.client.search(index=index, body=body)

    def get_document(self, index, id):
        return self.client.get(index=index, id=id)

    def iterdir(self):

        if not self.is_dir():
            return

        if self.path_type == 'root':
            for index in self.client.indices.get('*'):
                yield self.gen(f"/{index}")
        elif self.path_type == 'index':
            for doc in self._search.scan():
                yield self.gen(f"/{self.index}/{doc.meta.id}")
        else:
            for doc in self._search.query(self.query).scan():
                yield self.gen(f"/{self.index}/{doc.meta.id}")


    def read(self):
