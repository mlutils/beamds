import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from elasticsearch_dsl import Search, Q, DenseVector, SparseVector, Document, Index, Text
from elasticsearch_dsl.query import Query
from datetime import datetime

from ..path import PureBeamPath, normalize_host
from ..utils import lazy_property as cached_property, recursive_elementwise
from ..type import check_type

from .core import BeamDoc
from .utils import parse_kql_to_dsl, generate_document_class



class BeamElastic(PureBeamPath, BeamDoc):

    def __init__(self, *args, hostname=None, port=None, username=None, password=None, verify=False,
                 tls=False, client=None, keep_alive='1m', document=None, **kwargs):
        super().__init__(*args, hostname=hostname, port=port, username=username, password=password,
                         scheme='elastic', **kwargs)
        self.verify = verify
        self.tls = tls
        self.client = client or self._get_client()
        self.keep_alive = keep_alive
        self._values = None
        self._metadata = None
        self._doc_cls: Document | None = document
        self._index: Index | None = None

    # elasticsearch timestamp format with timezone
    timestamp_format = '%Y-%m-%dT%H:%M:%S%z'

    @staticmethod
    def to_datetime(timestamp: str | datetime) -> datetime:
        if isinstance(timestamp, datetime):
            return timestamp
        return datetime.strptime(timestamp, BeamElastic.timestamp_format)

    @staticmethod
    def to_timestamp(timestamp: str | datetime) -> str:
        if isinstance(timestamp, str):
            return timestamp
        return timestamp.strftime(BeamElastic.timestamp_format)

    def _get_client(self):
        protocol = 'https' if self.tls else 'http'
        host = f"{protocol}://{normalize_host(self.hostname, self.port)}"

        if (self.username, self.password) != (None, None):
            auth = (self.username, self.password)
        else:
            auth = None

        return Elasticsearch([host], http_auth=auth, verify_certs=self.verify)

    def _search_index(self, index):
        return Search(using=self.client, index=index)

    @property
    def index_name(self):
        if self.path_type in ['index', 'query', 'document']:
            return self.parts[1]
        return None

    @property
    def index(self) -> Index:
        if self._index is None:
            self._index = Index(self.index_name, using=self.client)
        return self._index

    @property
    def _search(self):
        return self._search_index(self.index_name)

    @property
    def q(self):

        if self.path_type in ['root', 'index']:
            return BeamElastic.match_all

        qs = self.parts[2:-1]
        q = BeamElastic.match_all

        for part in qs:
            q = q & parse_kql_to_dsl(part)

        if self.path_type == 'document':
            q = q & Q('ids', values=[self.parts[-1]])
        else:
            q = q & parse_kql_to_dsl(self.parts[-1])

        return q

    @property
    def s(self):
        if self.path_type in ['root', 'index']:
            return self._search
        return self._search.query(self.q)

    @property
    def path_type(self):
        if len(self.parts) == 1:
            return 'root'
        elif len(self.parts) == 2:
            return 'index'
        elif self._is_file_path:
            return 'document'
        else:
            return 'query'

    def get_document_class(self):
        if self.path_type == 'root':
            return None
        if self._doc_cls is None:
            self._doc_cls = generate_document_class(self.client, self.index_name)
        return self._doc_cls

    @property
    def document_class(self):
        return self.get_document_class()

    def set_document_class(self, doc_cls):
        if self.path_type == 'root':
            return ValueError("Cannot set document class for root path")
        self._doc_cls = doc_cls
        self.index.document(doc_cls)

    def _init_index(self):
        if self.document_class is not None:
            self.document_class.init(using=self.client)
        else:
            self.index.create(using=self.client)

    def _delete_index(self):
        if self.document_class is not None:
            self.document_class.delete(using=self.client)
        else:
            self.index.delete(using=self.client)

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

    def gen(self, path):
        PathType = type(self)
        return PathType(path, client=self.client, hostname=self.hostname, port=self.port, username=self.username,
                        password=self.password, fragment=self.fragment, params=self.params, document=self._doc_cls,
                        **self.query)

    def index_exists(self, index):
        return self.client.indices.exists(index=index)

    def create_index(self, index, body):
        return self.client.indices.create(index=index, body=body)

    def delete_index(self, index):
        return self.client.indices.delete(index=index)

    def index_document(self, index, body, id=None):
        return self.client.index(index=index, body=body, id=id)

    def index_bulk(self, index, docs):
        return bulk(self.client, docs, index=index)

    def search(self, index, body):
        return self.client.search(index=index, body=body)

    def get_document(self, index, id):
        return self.client.get(index=index, id=id)

    def mkdir(self, *args, **kwargs):
        if self.path_type in ['root', 'document']:
            raise ValueError("Cannot create root path")
        if self.path_type in ['index', 'query']:
            self.index.create(using=self.client)

    def rmdir(self, *args, **kwargs):
        if self.path_type in ['root', 'document']:
            raise ValueError("Cannot delete root path")
        if self.path_type in ['index', 'query']:
            self.index.delete(using=self.client)

    def exists(self):
        if self.path_type == 'root':
            return bool(self.client.ping())
        if self.path_type == 'index':
            return self.index.exists(using=self.client)
        if self.path_type == 'document':
            return self.client.exists(index=self.index_name, id=self.parts[-1])
        # if it is a query type, check that at least one document matches the query
        s = self.s.extra(terminate_after=1)
        return s.execute().hits.total.value > 0

    def __len__(self):
        if self.path_type == 'root':
            return len(self.client.indices.get('*'))
        return self.s.count()

    def create_vector_search_index(self, index_name, field_name, dims=32, **other_fields):

        class SimpleVectorDocument(Document):
            vector = DenseVector(dims=dims)
            for field, field_type in other_fields.items():
                locals()[field] = field_type


    def iterdir(self):

        if not self.is_dir():
            return

        if self.path_type == 'root':
            for index in self.client.indices.get('*'):
                yield self.gen(f"/{index}")
        else:
            s = self.s.source(False)
            for doc in s.iterate(keep_alive=self.keep_alive):
                yield self.gen(f"/{self.index_name}/{doc.meta.id}")

    @property
    def values(self):
        if self.path_type == 'root':
            return list(self.client.indices.get('*'))
        else:
            return self._get_values()

    def _get_values(self):
        if self._values is None:
            self._values, self._metadata = self._get_values_and_metadata()
        return self._values

    def _get_metadata(self):
        if self._metadata is None:
            self._metadata = self._get_values_and_metadata(source=False)[1]
        return self._metadata

    def _get_values_and_metadata(self, source=True):
        v = []
        meta = []
        s = self.s if source else self.s.source(False)
        for doc in s.iterate(keep_alive=self.keep_alive):
            v.append(doc.to_dict())
            meta.append(doc.meta.to_dict())
        return v, meta

    def ping(self):
        return self.client.ping()

    @cached_property
    def df(self):
        import pandas as pd
        return pd.DataFrame(self.values)

    def add(self, x):
        if self.path_type == 'index':
            self.index_document(self.index_name, x)
        else:
            raise ValueError("Cannot add document to non-index path")

    def write_bulk(self, docs):
        if self.path_type == 'index':
            self.index_bulk(self.index_name, docs)
        else:
            raise ValueError("Cannot write bulk to non-index path")

    def write(self, *args, ext=None, **kwargs):

        for x in args:
            x_type = check_type(x)
            if x_type.minor == 'pandas':
                docs = x.to_dict(orient='records')
                self.write_bulk(docs)
            elif x_type.minor == 'dict':
                self.add(x)
            elif isinstance(x, Document):
                x.save(using=self.client, index=self.index_name)
            else:
                raise ValueError(f"Cannot write object of type {x_type}")

    def init(self):
        if self.path_type in ['index', 'query']:
            self._init_index()
        else:
            raise ValueError("Cannot init document or root path")

    def get_schema(self, index_name):
        s = self.client.indices.get_mapping(index=index_name)
        return dict(s)[index_name]['mappings']['properties']

    def _search_native(self, index, query: dict | Query, sort=None, size=None, search_after=None):

        if size is None:
            size = 1000

        if isinstance(query, Query):
            query = query.to_dict()

        while True:
            res = self.client.search(index=index, query=query, sort=sort, size=size, search_after=search_after)
            hits = res['hits']['hits']
            h = None
            for h in hits:
                yield h
            if h is None:
                break
            if sort is not None:
                search_after = h['sort']
            else:
                if size is not None:
                    total = res['hits']['total']['value']
                    if total >= size:
                        pass
                        # logger.warning(f"Total hits: {total}, returning only {size}")
                break

    def _write_df(self, df, index):

        @recursive_elementwise
        def clearn_none(x):
            if pd.isna(x):
                return None
            return x

        data = df.to_dict(orient='records')
        data = clearn_none(data)
        if '_id' in df.columns:
            for d in data:
                if pd.isna(d['_id']):
                    d.pop('_id')

        self.index_bulk(index, data)

    def _delete_docs(self, ids, index):
        actions = [{
            '_op_type': 'delete',
            '_index': index,
            '_id': i
        } for i in ids]
        bulk(self.client, actions)

    def query_df(self, query, index, sort=None, size=None, search_after=None, **kwargs):
        if size is None:
            size = 1000

        l = []
        i = []
        for r in self._search_native(index, query, sort=sort, size=size, search_after=search_after):
            l.append(r['_source'])
            i.append(r['_id'])

        df = pd.DataFrame(l, index=i)
        df.index.name = '_id'
        return df

    # def search(self, x, k=10, **kwargs):
    #     # use knn search to find similar documents kwargs is assumed to be a list of terms to filter the search
    #     q = Q('knn', **x)




