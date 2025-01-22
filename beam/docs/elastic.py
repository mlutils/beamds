import numpy as np
import pandas as pd
import torch
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, scan, BulkIndexError
from elasticsearch_dsl import Search, Q, DenseVector, SparseVector, Document, Index, Text, A
from elasticsearch_dsl.query import Query, Term
from datetime import datetime

from ..path import PureBeamPath, normalize_host
from ..utils import lazy_property as cached_property, recursive_elementwise
from ..type import check_type, Types
from ..utils import divide_chunks, retry

from .core import BeamDoc
from .utils import parse_kql_to_dsl, generate_document_class
from ..base import Loc, Groups



class BeamElastic(PureBeamPath, BeamDoc):

    '''
    Higher level (external) API methods:

    read() - read data from the index, given the path and its queries
    write() - write many types of data to the index
    # add() - add a document to the index
    delete() - delete the queried data from the index
    '''

    def __init__(self, *args, hostname=None, port=None, username=None, password=None, verify=False,
                 tls=False, client=None, keep_alive=None, sleep=None, document=None, q=None, max_actions=None, retries=None,
                 fragment=None, maximum_bucket_limit=None, fields=None, **kwargs):
        super().__init__(*args, hostname=hostname, port=port, username=username, password=password, tls=tls,
                         keep_alive=keep_alive, scheme='elastic', max_actions=max_actions, retries=retries,
                         sleep=sleep, fragment=fragment, maximum_bucket_limit=maximum_bucket_limit, **kwargs)

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
        self._q: Query | None = self.parse_query(q)

        if fields is not None:
            fields = fields if isinstance(fields, list) else [fields]
        else:
            fields = []

        more_fields = self.fragment.split(',') if bool(self.fragment) else []
        fields = list(set(fields + more_fields))

        # concatenate fields from more_fields and fields
        self.fields = fields if fields else None

        if max_actions is None:
            max_actions = 10000
        self.max_actions = int(max_actions)

        if retries is None:
            retries = 3
        self.retries = int(retries)

        if sleep is None:
            sleep = 0.1
        self.sleep = int(sleep)

        if maximum_bucket_limit is None:
            maximum_bucket_limit = 10000
        self.maximum_bucket_limit = int(maximum_bucket_limit)

        # self.groupby = Groups(self)
        self.loc = Loc(self)

    # elasticsearch timestamp format with timezone
    timestamp_format = '%Y-%m-%dT%H:%M:%S%z'

    def set_fields(self, f: str | list):
        f = f if isinstance(f, list) else [f]
        self.fields = f if self.fields is None else self.fields + f
        self.url.replace_fragment(','.join(self.fields))

    @staticmethod
    def parse_query(query: str | dict | Query | None) -> Query | None:
        if isinstance(query, Query):
            return query
        if isinstance(query, dict):
            return Q(query)
        if isinstance(query, str):
            return parse_kql_to_dsl(query)
        if query is None:
            return None
        raise ValueError("Invalid query type")

    def __repr__(self):
        return f"{str(self.url)} | {self.q}"

    @property
    def q(self):
        if self._q is None:
            return self.document_query
        return self._q & self.document_query

    def has_query(self):
        return self._q is not None

    def _assert_other_type(self, other):
        assert other.level == 'query', "Cannot combine with non-query path"
        assert self.index_name == other.index_name, "Cannot combine queries from different indices"
        assert self.client == other.client, "Cannot combine queries from different clients"

    def with_query(self, query):
        return self & query

    def __and__(self, other):
        q = self.q or BeamElastic.match_all()
        if type(other) is BeamElastic:
            self._assert_other_type(other)
            query = q & other.q
        else:
            query = q & self.parse_query(other)
        return self.gen(self.path, q=query)

    def __or__(self, other):
        q = self.q or BeamElastic.match_none()
        if type(other) is BeamElastic:
            self._assert_other_type(other)
            query = q | other.q
        else:
            query = q | self.parse_query(other)
        return self.gen(self.path, q=query)

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

    @property
    def document_id(self):
        if self.level == 'document':
            return self.parts[-1]
        return None

    @property
    def document_query(self):
        if self.level == 'document':
            return Q('ids', values=[self.document_id])
        return BeamElastic.match_all()

    @property
    def s(self):
        if self.level == 'root':
            return Search(using=self.client).source(self.fields).params(size=self.max_actions)
        if self.level == 'index':
            return self.index.search().source(self.fields).params(size=self.max_actions)
        return self.index.search().query(self.q).params(size=self.max_actions).source(self.fields)

    @property
    def level(self):
        if len(self.parts) == 1:
            return 'root'
        elif len(self.parts) == 2 and not self.has_query():
            return 'index'
        elif len(self.parts) == 3:
            return 'document'
        else:
            return 'query'

    def get_document_class(self):
        if self.level == 'root':
            return None
        if self._doc_cls is None and self._index_exists(self.index_name):
            self._doc_cls = generate_document_class(self.client, self.index_name)
        return self._doc_cls

    @property
    def document_class(self):
        return self.get_document_class()

    def set_document_class(self, doc_cls):
        if self.level == 'root':
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

    @staticmethod
    def match_all():
        return Q('match_all')

    @staticmethod
    def match_none():
        return Q('match_none')

    def gen(self, path, q=None, **kwargs):
        hostname = kwargs.pop('hostname', self.hostname)
        port = kwargs.pop('port', self.port)
        username = kwargs.pop('username', self.username)
        password = kwargs.pop('password', self.password)
        fragment = kwargs.pop('fragment', self.fragment)
        params = kwargs.pop('params', self.params)
        doc_cls = kwargs.pop('document', self._doc_cls)
        query = kwargs.pop('query', {})
        query = {**query, **kwargs}
        PathType = type(self)
        return PathType(path, client=self.client, hostname=hostname, port=port, username=username,
                        password=password, fragment=fragment, params=params, document=doc_cls, q=q, **query)

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
        if self.level == 'index':
            self._delete_index(self.index_name)
        elif self.level == 'document':
            self._delete_document(self.index_name, self.document_id)
        elif self.level == 'query':
            self.s.delete()
        else:
            raise ValueError("Cannot delete root path")

    def mkdir(self, *args, **kwargs):
        if self.level in ['root', 'document']:
            raise ValueError("Cannot create root path")
        if self.level in ['index', 'query']:
            self.index.create(using=self.client)

    def rmdir(self, *args, **kwargs):
        if self.level in ['root', 'document']:
            raise ValueError("Cannot delete root path")
        if self.level in ['index', 'query']:
            self.index.delete(using=self.client)

    def exists(self):
        if self.level == 'root':
            return bool(self.client.ping())
        if self.level == 'index':
            return self.index.exists(using=self.client)
        if self.level == 'document':
            return self.client.exists(index=self.index_name, id=self.parts[-1])
        # if it is a query type, check that at least one document matches the query
        s = self.s.extra(terminate_after=1)
        return s.execute().hits.total.value > 0

    def __len__(self):
        return self.count()

    def create_vector_search_index(self, index_name, field_name, dims=32, **other_fields):

        class SimpleVectorDocument(Document):
            vector = DenseVector(dims=dims)
            for field, field_type in other_fields.items():
                locals()[field] = field_type

    def iterdir(self, wildcard=None, alias=True, hidden=False, alias_only=False):

        wildcard = wildcard or '*'


        if self.level == 'root':

            if not alias_only:
                for index in self.client.indices.get(index=wildcard):
                    if index.startswith('.') and not hidden:
                        continue
                    yield self.gen(f"/{index}")

            if alias:
                for alias in self.client.indices.get_alias(wildcard).keys():
                    yield self.gen(f"/{alias}")

        else:
            s = self.s.source(False)
            for doc in s.iterate(keep_alive=self.keep_alive):
                yield self.gen(f"/{self.index_name}/{doc.meta.id}")

    @property
    def values(self):
        if self.level == 'root':
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

    def items(self):
        if self.level == 'root':
            for index in self.client.indices.get('*'):
                yield index, self.gen(f"/{index}")
        else:
            for doc in self.s.iterate(keep_alive=self.keep_alive):
                yield doc.meta.id, doc.to_dict()

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
                self._index_document(self.index_name, x, id=ids)
            elif isinstance(x, Document):
                x.save(using=self.client, index=self.index_name)
            else:
                raise ValueError(f"Cannot write object of type {x_type}")

    def read(self, as_df=False, as_dict=False, as_iter=True, source=True):

        if self.level == 'document':
            doc = self._get_document(self.index_name, self.document_id)
            if as_dict:
                return doc['_source']
            return doc

        if as_df:
            return pd.DataFrame(self.values)

        if as_dict:
            return self.values

        if as_iter:
            for doc in self.s.iterate(keep_alive=self.keep_alive):
                yield doc

    def init(self):
        if self.level in ['index', 'query']:
            self._init_index()
        else:
            raise ValueError("Cannot init document or root path")

    def get_schema(self, index_name):
        s = self.client.indices.get_mapping(index=index_name)
        return dict(s)[index_name]['mappings']['properties']

    @cached_property
    def schema(self):
        return self.get_schema(self.index_name)

    @staticmethod
    @recursive_elementwise
    def _clear_none(x):
        if pd.isna(x):
            return None
        return x

    def sanitize_input(self, x):
        x = self._clear_none(x)
        return x

    def count(self):
        if self.level == 'root':
            return len(self.client.indices.get('*'))
        elif self.level == 'index':
            return self.client.count(index=self.index_name)['count']
        elif self.level == 'document':
            return 1
        else:
            return self.s.count()

    def unlink(self, **kwargs):
        return self.delete()

    def get_vector_field(self):
        schema = self.schema
        for field, field_type in schema.items():
            if field_type['type'] == 'dense_vector':
                return field
        raise ValueError(f"No dense vector field found in schema for index {self.index_name}")

    # search knn query
    def search(self, x: np.ndarray | list | torch.Tensor, k=10, field=None):

        if isinstance(x, np.ndarray):
            x = x.tolist()
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy().tolist()

        if field is None:
            self.get_vector_field()

        q = Q('knn', field=field, vector=x)
        s = self.s.query(q).extra(size=k)
        return s.execute()

    def unique(self, field_name, size=None):

        field_name = self.keyword_field(field_name)
        s = self.s.source([field_name])

        if size is not None and size <= self.maximum_bucket_limit:
            terms_agg = A("terms", field=field_name, size=size)  # Increase size for more unique values
            s.aggs.bucket("unique_values", terms_agg)

            # Execute the search
            response = s.execute()

            # Retrieve the unique values
            unique_values = [bucket.key for bucket in response.aggregations.unique_values.buckets]

        else:

            unique_values = self.value_counts(field_name).index.tolist()

        return unique_values

    def keyword_field(self, field_name):

        schema = self.schema
        if field_name in schema:
            if 'fields' in schema[field_name] and 'keyword' in schema[field_name]['fields']:
                return f"{field_name}.keyword"

        return field_name

    def nunique(self, field_name=None):

        s = self.s
        field_name = self.get_unique_field(field_name)

        cardinality_agg = A("cardinality", field=field_name)
        s.aggs.metric("unique_count", cardinality_agg)

        # Execute the search
        response = s.execute()

        # Retrieve the count of unique values
        unique_count = response.aggregations.unique_count.value
        return unique_count

    def is_date_field(self, field_name):
        schema = self.schema
        if field_name not in schema:
            return False
        return schema[field_name]['type'] == 'date'

    def get_unique_field(self, field_name=None, as_keyword=True):

        if field_name is None:
            if len(self.fields) == 1:
                field_name = self.fields[0]
            else:
                raise ValueError("Cannot infer field name from multiple fields object")

        field_name = self.keyword_field(field_name)

        if as_keyword:
            field_name = self.keyword_field(field_name)

        return field_name


    def value_counts(self, field_name=None, sort=True, normalize=False):

        # Execute the search and paginate
        counts = {}
        after_key = None  # Initialize the after_key
        field_name = self.get_unique_field(field_name)

        while True:

            composite_kwargs = dict(sources=[{"unique_values": {"terms": {"field": field_name}}}],
                                    size=self.maximum_bucket_limit, )  # Maximum allowed size per request

            if after_key:
                composite_kwargs["after"] = after_key

            # Use terms aggregation
            composite_agg = A("composite", **composite_kwargs)
            s = self.s
            s.aggs.bucket("unique_values", composite_agg)

            response = s.execute()
            buckets = response.aggregations.unique_values.buckets
            counts.update({bucket.key.unique_values: bucket.doc_count for bucket in buckets})

            if len(buckets) < self.maximum_bucket_limit:
                break
            else:
                after_key = response.aggregations.unique_values.after_key

        if self.is_date_field(field_name):
            counts = {pd.to_datetime(k, unit="ms"): v for k, v in counts.items()}

        c = pd.Series(counts)
        if sort:
            c = c.sort_values(ascending=False)
        if normalize:
            c = c / c.sum()
        return c

    def __getitem__(self, item):

        if self.level == 'root':
            return self.gen(f"/{item}")
        else:
            fields = [item] if isinstance(item, str) else item
            if self.fields is not None:
                if set(fields) - set(self.fields):
                    raise ValueError(f"Cannot select fields {list(set(fields) - set(self.fields))} not in {self.fields}")
            return self.gen(self.path, fields=fields)

    def _loc(self, ind):

        if isinstance(ind, str):
            return self.joinpath(ind)
        else:
            q = Q('ids', values=ind)
            return self & q


    def add_alias(self, alias_name, routing=None, **kwargs):

        if self.level not in ['index', 'query']:
            raise ValueError("Cannot add alias to non-index/query paths")

        body = kwargs
        if routing is not None:
            body['routing'] = routing

        if self.q is not None:
            body['query'] = self.q.to_dict()

        return self.client.indices.put_alias(index=self.index_name, name=alias_name, body=body)

    def remove_alias(self, alias_name, **kwargs):

        if self.level not in ['index', 'query']:
            raise ValueError("Cannot remove alias from non-index/query paths")

        return self.client.indices.delete_alias(index=self.index_name, name=alias_name, **kwargs)

    def reindex(self, target_index: str, **kwargs):
        return self.client.reindex(body={"source": {"index": self.index_name},
                                         "dest": {"index": target_index.index_name}}, **kwargs)

    def copy(self, path, **kwargs):
        # use reindex to copy data from one index to another
        if type(path) is BeamElastic:
            target_index = path.index_name
        else:
            target_index = path

        return self.reindex(target_index, **kwargs)


    def term(self, field, value):
        return Q('term', **{self.keyword_field(field): value})
