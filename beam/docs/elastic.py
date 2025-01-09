import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, scan, BulkIndexError
from elasticsearch_dsl import Search, Q, DenseVector, SparseVector, Document, Index, Text
from elasticsearch_dsl.query import Query
from datetime import datetime

from ..path import PureBeamPath, normalize_host
from ..utils import lazy_property as cached_property, recursive_elementwise
from ..type import check_type, Types
from ..utils import divide_chunks, retry

from .core import BeamDoc
from .utils import parse_kql_to_dsl, generate_document_class



class BeamElastic(PureBeamPath, BeamDoc):

    '''
    Higher level (external) API methods:

    read() - read data from the index, given the path and its queries
    write() - write many types of data to the index
    # add() - add a document to the index
    delete() - delete the queried data from the index
    '''

    def __init__(self, *args, hostname=None, port=None, username=None, password=None, verify=False,
                 tls=False, client=None, keep_alive=None, sleep=None, document=None, eql=None, max_actions=None, retries=None,
                 **kwargs):
        super().__init__(*args, hostname=hostname, port=port, username=username, password=password, tls=tls,
                         keep_alive=keep_alive, scheme='elastic', max_actions=max_actions, retries=retries,
                         sleep=sleep, **kwargs)

        self.verify = verify

        if type(tls) is str:
            tls = tls.lower() == 'true'
        self.tls = tls

        self.client = client or self._get_client()
        self.keep_alive = keep_alive or '1m'
        self._values = None
        self._metadata = None
        self._doc_cls: Document | None = document
        self._index: Index | None = None
        self._eql: Query = eql or None

        if max_actions is None:
            max_actions = 1000
        self.max_actions = int(max_actions)

        if retries is None:
            retries = 3
        self.retries = int(retries)

        if sleep is None:
            sleep = 0.1
        self.sleep = int(sleep)

    # elasticsearch timestamp format with timezone
    timestamp_format = '%Y-%m-%dT%H:%M:%S%z'

    @property
    def dsl(self):
        return self._eql or BeamElastic.match_all

    def __and__(self, other):
        if type(other) is Query:
            query = self.dsl & other
        elif type(other) is BeamElastic:
            query = self.dsl & other.dsl
        else:
            raise ValueError(f"Cannot combine {type(other)} with {type(self)}")
        return self.gen(self.path, eql=query)

    def __or__(self, other):
        if type(other) is Query:
            q = other
        elif type(other) is BeamElastic:
            q = other.dsl
        else:
            raise ValueError(f"Cannot combine {type(other)} with {type(self)}")

        if self._eql is None:
            query = q
        else:
            query = self.dsl | q

        return self.gen(self.path, eql=query)

    @property
    def full_query(self):
        return self.kql & self.dsl

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

    @property
    def index_name(self):
        if len(self.parts) > 1:
            return self.parts[1]
        return None

    @property
    def index(self) -> Index:
        if self._index is None:
            self._index = Index(self.index_name, using=self.client)
        return self._index

    def search(self, query=None, as_df=False, as_dict=False, as_iter=True):

        if query is None:
            query = self.full_query

        s = self.index.search().query(query)

        if as_df:
            return pd.DataFrame([doc.to_dict() for doc in s.scan()])
        if as_dict:
            return [doc.to_dict() for doc in s.scan()]
        if as_iter:
            for doc in s.scan():
                yield doc.to_dict()

    @property
    def kql(self):

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
            return self.index.search()
        return self.index.search().query(self.full_query)

    @property
    def path_type(self):
        if len(self.parts) == 1:
            return 'root'
        elif len(self.parts) == 2 and self._eql is None:
            return 'index'
        elif self._is_file_path:
            return 'document'
        else:
            return 'query'

    def get_document_class(self):
        if self.path_type == 'root':
            return None
        if self._doc_cls is None and self._index_exists(self.index_name):
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
        if not self.index.exists():
            if self.document_class is not None:
                self.document_class.init(index=self.index_name, using=self.client)
            else:
                self.index.create(using=self.client)

    def _delete_index2(self):
        if self.document_class is not None:
            self.document_class.delete(using=self.client)
        else:
            self.index.delete(using=self.client)

    @property
    def _is_file_path(self):
        # check if last part of the path is _id value
        p = self.parts[-1]
        return ':' not in p and '.' not in p and len(self.parts) > 2

    @staticmethod
    @property
    def match_all():
        return Q('match_all')

    @staticmethod
    @property
    def match_none():
        return Q('match_none')

    def gen(self, path, eql=None):
        PathType = type(self)
        return PathType(path, client=self.client, hostname=self.hostname, port=self.port, username=self.username,
                        password=self.password, fragment=self.fragment, params=self.params, document=self._doc_cls,
                        eql=eql, **self.query)

    # list of native api methods
    def _index_exists(self, index_name):
        return self.client.indices.exists(index=index_name)

    def _create_index(self, index_name, body):
        return self.client.indices.create(index=index_name, body=body)

    def _delete_index(self, index_name):
        return self.client.indices.delete(index=index_name)

    def _index_document(self, index_name, body, id=None):
        return self.client.index(index=index_name, body=body, id=id)

    def _index_bulk(self, index_name, docs, ids=None, sanitize=False):

        if sanitize:
            docs = self.sanitize_input(docs)

        if ids is None:
            actions = [{"_index": index_name, "_source": doc} for doc in docs]
        else:
            actions = [{"_index": index_name, "_source": doc, "_id": i} for doc, i in zip(docs, ids)]

        retry_bulk = retry(func=bulk, retries=self.retries, logger=None, name=None, verbose=False, sleep=1)

        for i, c in divide_chunks(actions, chunksize=self.max_actions, chunksize_policy='ceil'):
            try:
                retry_bulk(self.client, c)
            except BulkIndexError as e:
                print(f"{len(e.errors)} document(s) failed to index.")
                for error in e.errors:
                    print("Error details:", error)
                raise e

    def _search_index(self, index_name, body):
        return self.client.search(index=index_name, body=body)

    def _delete_by_query(self, index_name, body):
        return self.client.delete_by_query(index=index_name, body=body)

    def _delete_document(self, index_name, id):
        return self.client.delete(index=index_name, id=id)

    def _get_document(self, index_name, id):
        return self.client.get(index=index_name, id=id)

    def delete(self):
        if self.path_type == 'index':
            self._delete_index(self.index_name)
        elif self.path_type == 'document':
            self._delete_document(self.index_name, self.parts[-1])
        elif self.path_type == 'query':
            self._delete_by_query(self.index_name, self.kql.to_dict())
        else:
            raise ValueError("Cannot delete root path")

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

    def as_df(self):
        import pandas as pd
        return pd.DataFrame(self.values)

    def add(self, x):
        if self.path_type == 'index':
            self._index_document(self.index_name, x)
        else:
            raise ValueError("Cannot add document to non-index path")

    def write(self, *args, ids=None, sanitize=False, **kwargs):

        for x in args:
            x_type = check_type(x)
            if x_type.minor in [Types.pandas, Types.cudf, Types.polars]:
                if x_type.minor != Types.pandas:
                    x = x.to_pandas()
                docs = x.to_dict(orient='records')
                if ids is True:
                    ids = x.index.tolist()
                self._index_bulk(self.index_name, docs, ids=ids, sanitize=sanitize)
            elif x_type.minor == Types.list:
                self._index_bulk(self.index_name, x, sanitize=sanitize)
            elif x_type.minor == Types.dict:
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

    @staticmethod
    @recursive_elementwise
    def _clear_none(x):
        if pd.isna(x):
            return None
        return x

    def sanitize_input(self, x):
        x = self._clear_none(x)
        return x

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

        self._index_bulk(index, data)

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

    def count(self):
        if self.path_type == 'root':
            return len(self.client.indices.get('*'))
        elif self.path_type == 'index':
            return self.client.count(index=self.index_name)['count']
        elif self.path_type == 'document':
            return 1
        else:
            return self.s.count()


    # def search(self, x, k=10, **kwargs):
    #     # use knn search to find similar documents kwargs is assumed to be a list of terms to filter the search
    #     q = Q('knn', **x)
