import json
from argparse import Namespace

import numpy as np
import pandas as pd

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, scan, BulkIndexError
from elasticsearch_dsl import Search, Q, DenseVector, SparseVector, Document, Index, Text, A
from elasticsearch_dsl.query import Query, Term
from datetime import datetime
from pydantic import BaseModel

from ..path import PureBeamPath, normalize_host
from ..utils import lazy_property as cached_property, recursive_elementwise
from ..type import check_type, Types
from ..utils import divide_chunks, retry
from ..importer import torch

from .core import BeamDoc
from .utils import parse_kql_to_dsl, generate_document_class, describe_dataframe
from ..base import Loc
from ..llm import beam_llm
from .queries import TimeFilter
from .groupby import Groupby


class LLMQueryResponse(BaseModel):
    index: str
    query_json_format: str
    short_description: str


class BeamElastic(PureBeamPath, BeamDoc):

    '''
    Higher level (external) API methods:

    read() - read data from the index, given the path and its queries
    write() - write many types of data to the index
    # add() - add a document to the index
    delete() - delete the queried data from the index
    '''

    exclude_hidden_pattern = '-.*'
    date_format = '%Y-%m-%d %H:%M:%S'

    def __init__(self, *args, hostname=None, port=None, username=None, password=None, verify=False,
                 tls=False, client=None, keep_alive=None, sleep=None, document=None, q=None, max_actions=None, retries=None,
                 fragment=None, maximum_bucket_limit=None, fields=None, sort_by=None, llm=None, timeout=None,
                 **kwargs):
        super().__init__(*args, hostname=hostname, port=port, username=username, password=password, tls=tls,
                         keep_alive=keep_alive, scheme='elastic', max_actions=max_actions, retries=retries,
                         sleep=sleep, fragment=fragment, maximum_bucket_limit=maximum_bucket_limit, **kwargs)

        self.verify = verify

        if type(tls) is str:
            tls = tls.lower() == 'true'
        self.tls = tls
        self.timeout = float(timeout) if timeout is not None else None

        if retries is None:
            retries = 3
        self.retries = int(retries)

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
        self.sort_by = sort_by
        self.llm = beam_llm(llm)

        if max_actions is None:
            max_actions = 10000
        self.max_actions = int(max_actions)

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

        s = str(self.url)

        if self.q is not None:
            fixed_q = str(self.q)
            if len(fixed_q) > 50:
                fixed_q = fixed_q[:50] + "..."
            s = f"{s} | query: {fixed_q}"

        if self.fields:
            s += f" | fields: {self.fields}"

        if self.sort_by:
            s += f" | sort: {self.sort_by}"

        return s

    @property
    def q(self):
        if self.level == 'query':
            return self._q
        if self.level == 'document':
            return self.document_query
        return None

    def ask(self, question, llm=None, execute=False, answer=True, **kwargs):

        if llm is None:
            llm = self.llm

        if llm is None:
            raise ValueError("LLM resource not set")
        has_index = self.index_name is not None
        if has_index:
            index_string = f"Index: {self.index_name}\n"
        else:
            index_string = ""

        base_query = self.q
        if base_query is not None:
            query_string = f"A base query is already set: {base_query.to_dict()}, you should assume that your query will be added (with and) to this query\n"
        else:
            query_string = ""

        from ..llm.tools import LLMGuidance
        guidance = LLMGuidance(guided_json=LLMQueryResponse)
        prompt = (f"You are an agent that interacts with an ElasticSearch database. "
                  f"You are required to answer the users' questions based on the data in the database. "
                  f"You can generate a DSL query to retrieve the relevant data in order to answer the question.\n\n"
                  f"The dataset schema is:\n"
                  f"{index_string}"
                  f"{self.schema}\n\n"
                  f"{query_string}"
                  f"User's question: {question}\n\n"
                  f"Your response should follow the following JSON format:\n"
                  f"{LLMQueryResponse.model_json_schema()}")

        llm.reset_chat()
        res = llm.chat(prompt, guidance=guidance, **kwargs).json

        query = json.loads(res['query_json_format'])
        description = res['short_description']
        index = res['index']

        fields = None
        sort = None
        if 'query' in query:
            q = Q(query['query'])
            if '_source' in query:
                fields = query['_source']
            if 'sort' in query:
                sort = query['sort']
        else:
            q = Q(query)
        df = None
        text_answer = None
        if execute:
            ind = self & q
            if fields is not None:
                ind.set_fields(fields)
            if sort is not None:
                ind.sort(sort)

            df = ind.as_df()
            if answer:

                if df is None or df.empty:
                    info = "No matching documents found."
                else:
                    info = describe_dataframe(df, n_samples=10)
                prompt = (f"Based on the data retrieved from the database:"
                          f"{info}\n\n"
                          f" please provide a text answer to the user's question: {question}\n\n")

                text_answer = llm.chat(prompt, **kwargs).text

        return Namespace(query=q, description=description, index=index, df=df, text_answer=text_answer)

    def has_query(self):
        return self._q is not None

    def _assert_other_type(self, other):
        assert other.level == 'query', "Cannot combine with non-query path"
        assert self.index_name == other.index_name, "Cannot combine queries from different indices"
        assert self.client == other.client, "Cannot combine queries from different clients"

    def with_query(self, query):
        return self & query

    def __and__(self, other):
        q = self.q
        if type(other) is BeamElastic:
            self._assert_other_type(other)
            query = other.q
        else:
            query = self.parse_query(other)

        if q is not None:
            query = q & query

        return self.gen(self.path, q=query)

    def __or__(self, other):
        q = self.q
        if type(other) is BeamElastic:
            self._assert_other_type(other)
            query = other.q
        else:
            query = self.parse_query(other)

        if q is not None:
            query = q | query

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

        kwargs = {}
        if self.timeout is not None:
            kwargs['request_timeout'] = self.timeout

        return Elasticsearch([host], http_auth=auth, verify_certs=self.verify, max_retries=self.retries,  # Number of retries
                             retry_on_status={500, 502, 503, 504},  # Retry on these errors
                             retry_on_timeout=True)  # Retry when a timeout occurs

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
        return None

    @property
    def s(self):
        if self.level == 'root':
            return Search(using=self.client).source(self.fields).params(size=self.max_actions)
        if self.level == 'index':
            s = self.index.search().source(self.fields).params(size=self.max_actions)
        else:
            s = self._s.params(size=self.max_actions).source(self.fields)
        if self.sort_by is not None:
            s = s.sort(self.sort_by)
        return s

    @property
    def _s(self):
        if self.level == 'root':
            return Search(using=self.client)
        elif self.level == 'index':
            return self.index.search()
        return self.index.search().query(self.q)

    def sort(self, field):
        return self.gen(self.path, sort_by=field)

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

    def gen(self, path, **kwargs):
        hostname = kwargs.pop('hostname', self.hostname)
        port = kwargs.pop('port', self.port)
        username = kwargs.pop('username', self.username)
        password = kwargs.pop('password', self.password)
        fragment = kwargs.pop('fragment', self.fragment)
        params = kwargs.pop('params', self.params)
        doc_cls = kwargs.pop('document', self._doc_cls)
        query = kwargs.pop('query', {})
        fields = kwargs.pop('fields', self.fields)
        sort_by = kwargs.pop('sort_by', self.sort_by)
        llm = kwargs.pop('llm', self.llm)
        q = kwargs.pop('q', self.q)

        # must be after extracting all other kwargs
        query = {**query, **kwargs}
        PathType = type(self)
        return PathType(path, client=self.client, hostname=hostname, port=port, username=username, fields=fields,
                        password=password, fragment=fragment, params=params, document=doc_cls, q=q, sort_by=sort_by,
                        llm=llm, **query)

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
            # if not exists
            if not self.index.exists():
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

                for ind, av in self.client.indices.get_alias(index=wildcard).items():
                    for a in av['aliases'].keys():
                        if a.startswith('.') and not hidden:
                            continue
                        yield self.gen(f"/{ind}/{a}")

        else:
            s = self.s.source(False)
            for doc in s.iterate(keep_alive=self.keep_alive):
                yield self.gen(f"/{self.index_name}/{doc.meta.id}")

    @property
    def values(self):
        if self.level == 'root':
            return list(self.client.indices.get('*'))
        else:
            return self._get_all_values()

    def _get_all_values(self, **kwargs):
        if self._values is None:
            self._values, self._metadata = self._get_values_and_metadata(**kwargs)
        return self._values

    def _get_all_metadata(self, **kwargs):
        if self._metadata is None:
            self._metadata = self._get_values_and_metadata(_source=False, **kwargs)[1]
        return self._metadata

    def _get_values_and_metadata(self, _source=True, size=None, pagination=None, add_score=False, **kwargs):
        v = []
        meta = []
        s = self.s if _source else self.s.source(False)
        if size is not None:
            s = s.params(size=size)

        if pagination is None:
            pagination = size is None or size > self.max_actions

        if pagination:
            iterator = enumerate(s.iterate(keep_alive=self.keep_alive))
        elif not pagination and add_score:
            iterator = enumerate(s.execute())
        else:
            iterator = enumerate(s.scan())
        for i, doc in iterator:
            if i == size:
                break
            v.append(doc.to_dict())
            meta.append(doc.meta.to_dict())
        return v, meta

    def get_values_and_metadata(self, size=None, **kwargs):
        if size is None:
            return self._get_all_values(**kwargs), self._get_all_metadata(**kwargs)
        return self._get_values_and_metadata(size=size, **kwargs)

    def ping(self):
        return self.client.ping()

    def as_df(self, add_ids=True, add_score=False, add_index_name=False, size=None, pagination=None, use_score=False):
        import pandas as pd
        return_score = add_score or use_score
        v, m = self.get_values_and_metadata(size=size, pagination=pagination, add_score=return_score)
        index = None
        if add_ids:
            index = [x['id'] for x in m]

        df = pd.DataFrame(v, index=index)
        if add_score:
            df['_score'] = [x['score'] for x in m]
        if add_index_name:
            df['_index_name'] = self.index_name

        return df

    def as_cudf(self, add_ids=True, add_score=False, add_index_name=False, size=None, pagination=None, use_score=False):
        import cudf
        return_score = add_score or use_score
        v, m = self.get_values_and_metadata(size=size, pagination=pagination, add_score=return_score)
        index = None
        if add_ids:
            index = [x['id'] for x in m]

        df = cudf.DataFrame(v, index=index)
        if add_score:
            df['_score'] = [x['score'] for x in m]
        if add_index_name:
            df['_index_name'] = self.index_name

        return df

    def as_pl(self, add_ids=True, add_score=False, add_index_name=False, size=None, pagination=None, use_score=False):
        import polars as pl
        return_score = add_score or use_score
        v, m = self.get_values_and_metadata(size=size, pagination=pagination, add_score=return_score)
        # Convert values to a Polars DataFrame
        df = pl.DataFrame(v)

        # Add index column if needed
        if add_ids:
            ids = [x['id'] for x in m]
            df = df.with_columns(pl.Series("_id", ids))
            # make the _id column the first column
            df = df.select(["_id"] + [c for c in df.columns if c != "_id"])

        # Add score column if requested
        if add_score:
            scores = [x['score'] for x in m]
            df = df.with_columns(pl.Series("_score", scores))

        # Add index name column if requested
        if add_index_name:
            df = df.with_columns(pl.lit(self.index_name).alias("_index_name"))

        return df

    def as_dict(self, add_metadata=False, size=None, pagination=None, add_score=False):
        v, m = self.get_values_and_metadata(size=size, pagination=pagination, add_score=add_score)
        if add_metadata:
            return v, m
        return v

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

    def read(self, as_df=False, as_dict=False, as_iter=True, source=True, add_ids=True, add_score=False,
             add_index_name=False, add_metadata=False, size=None):

        if self.level == 'document':
            doc = self._get_document(self.index_name, self.document_id)
            if as_dict:
                return doc['_source']
            return doc

        if as_df:
            return self.as_df(add_ids=add_ids, add_score=add_score, add_index_name=add_index_name, size=size)

        if as_dict:
            return self.as_dict(add_metadata=add_metadata, size=size)

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
        ind = self.index_name
        if ind is None:
            return self.get_schema(self.exclude_hidden_pattern)
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
            return len(list(self.client.indices.get('*')))
        elif self.level == 'index':
            return self.client.count(index=self.index_name)['count']
        elif self.level == 'document':
            return 1
        else:
            return self._s.count()

    def unlink(self, **kwargs):
        return self.delete()

    def get_vector_field(self):
        schema = self.schema
        for field, field_type in schema.items():
            if field_type['type'] == 'dense_vector':
                return field
        raise ValueError(f"No dense vector field found in schema for index {self.index_name}")

    # search knn query
    def search(self, x, k=10, field=None, num_candidates=1000):

        if isinstance(x, np.ndarray):
            x = x.tolist()
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy().tolist()

        if field is None:
            field = self.get_vector_field()

        knn = {"field": field, "vector": x, "k": k}
        c = self.client
        r = c.search(index=self.index_name, knn={"vector": x, "field": field, "k": k}, num_candidates=num_candidates)

        r = r['hits']['hits']
        return r

        # q = Q('knn', field=field, vector=x)
        # s = self.s.query(q).extra(size=k)
        # return s.execute()

    def unique(self, field_name=None, size=None):

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

    def agg(self, agg, field_name=None, **kwargs):

        s = self.s

        if self.fields is None or len(self.fields) == 1:
            field_name = self.get_unique_field(field_name)
            s.aggs.metric(agg, A(agg, field=field_name, **kwargs))
            response = s.execute()
            v = response.aggregations[agg].value
            v = self.format_aggregation_data(v, field_name, agg)
            return v

        else:

            for field in self.fields:
                s.aggs.metric(f"{agg}_{field}", A(agg, field=field, **kwargs))
            response = s.execute()
            v = {field: self.format_aggregation_data(response.aggregations[f"{agg}_{field}"].value,
                                                     field, agg) for field in self.fields}
            return pd.Series(v, name=agg)

    def format_aggregation_data(self, value, field, agg):
        if self.is_date_field(field) and agg in ['min', 'max', 'avg']:
            value = datetime.fromtimestamp(value / 1000)
        return value

    def mean(self, field_name=None):
        return self.agg('avg', field_name)

    def sum(self, field_name=None):
        return self.agg('sum', field_name)

    def min(self, field_name=None):
        return self.agg('min', field_name)

    def max(self, field_name=None):
        return self.agg('max', field_name)

    def median(self, field_name=None):
        return self.agg('percentiles', field_name)

    def std(self, field_name=None):
        return self.agg('std_deviation', field_name)

    def percentile(self, field_name=None, percentiles=(25, 50, 75)):
        percentiles = list(percentiles)
        return self.agg('percentiles', field_name, percents=percentiles)

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

    def copy(self, dst, **kwargs):

        if self.level in ['document', 'query']:

            if type(dst) is str:
                if '://' in dst:
                    from .resource import beam_path
                    dst = beam_path(dst)
                else:
                    dst = self.gen(dst)

            values = self.read(as_dict=True)
            dst.write(values, sanitize=True)

        elif self.level == 'index':

            # use reindex to copy data from one index to another
            if type(dst) is BeamElastic:
                target_index = dst.index_name
            else:
                target_index = dst

            return self.reindex(target_index, **kwargs)

        else:
            raise ValueError("Cannot copy from root path")

    def with_filter_term(self, value, field=None, as_keyword=True):
        return self & self.filter_term(value, field=field, as_keyword=as_keyword)

    def with_filter_terms(self, values, field=None, as_keyword=True):
        return self & self.filter_terms(values, field=field, as_keyword=as_keyword)

    def with_filter_time_range(self, field=None, start=None, end=None, period=None, pattern=None):
        return self & self.filter_time_range(field=field, start=start, end=end, period=period, pattern=pattern)

    def with_filter_whitelist(self, values, field=None, as_keyword=True):
        return self & self.filter_whitelist(values, field=field, as_keyword=as_keyword)

    def with_filter_blacklist(self, values, field=None, as_keyword=True):
        return self & self.filter_blacklist(values, field=field, as_keyword=as_keyword)

    def with_filter_gte(self, value, field=None):
        return self & self.filter_gte(value, field=field)

    def with_filter_gt(self, value, field=None):
        return self & self.filter_gt(value, field=field)

    def with_filter_lte(self, value, field=None):
        return self & self.filter_lte(value, field=field)

    def with_filter_lt(self, value, field=None):
        return self & self.filter_lt(value, field=field)

    def filter_term(self, value, field=None, as_keyword=True):
        field = self.get_unique_field(field, as_keyword=as_keyword)
        return  Q('term', **{self.keyword_field(field): value})

    def filter_terms(self, values, field=None, as_keyword=True):
        field = self.get_unique_field(field, as_keyword=as_keyword)
        return Q('terms', **{self.keyword_field(field): values})

    def filter_time_range(self, field=None, start=None, end=None, period=None, pattern=None):
        field = self.get_unique_field(field, as_keyword=False)
        return TimeFilter(field=field, start=start, end=end, period=period, pattern=pattern)

    def filter_whitelist(self, values, field=None, as_keyword=True):
        return self.filter_terms(values, field=field, as_keyword=as_keyword)

    def filter_blacklist(self, values, field=None, as_keyword=True):
        return ~self.filter_terms(values, field=field, as_keyword=as_keyword)

    def filter_gte(self, value, field=None):
        field = self.get_unique_field(field, as_keyword=False)
        return Q('range', **{field: {'gte': value}})

    def filter_gt(self, value, field=None):
        field = self.get_unique_field(field, as_keyword=False)
        return Q('range', **{field: {'gt': value}})

    def filter_lte(self, value, field=None):
        field = self.get_unique_field(field, as_keyword=False)
        return Q('range', **{field: {'lte': value}})

    def filter_lt(self, value, field=None):
        field = self.get_unique_field(field, as_keyword=False)
        return Q('range', **{field: {'lt': value}})

    def groupby(self, field_names, size=None, **kwargs):
        maximum_bucket_limit = size or self.maximum_bucket_limit
        return Groupby(self, field_names, size=maximum_bucket_limit, **kwargs)

    def __ge__(self, other):
        return self.filter_gte(other)

    def __gt__(self, other):
        return self.filter_gt(other)

    def __le__(self, other):
        return self.filter_lte(other)

    def __lt__(self, other):
        return self.filter_lt(other)

    def __eq__(self, other):
        return self.filter_term(other)

    def head(self, n=5):
        return self.as_df(size=n)

    def sort_values(self, field, ascending=True):
        return self & Q('sort', **{field: 'asc' if ascending else 'desc'})

    def random_generator(self, seed=None, field=None):
        if seed is None:
            seed = np.random.randint(2**32)
        rs = {'seed': seed}
        if field is not None:
            rs['field'] = self.keyword_field(field)

        q = Q('function_score', functions=[{'random_score': rs}])
        return self & q

    def get_best_field_to_randomize(self):
        """
        Determines the best field for use with `random_score`:
        1. Prefers a numeric field (integer, long, float) for stable randomization.
        2. Uses `_seq_no` if available (good for recent documents).
        3. Falls back to `_id` (random but not evenly distributed).
        4. Returns `None` if no suitable field is found.
        """
        schema = self.schema
        n = self.count()

        # 1️⃣ Prefer numeric fields (better for consistent randomization)
        for field, field_metadata in schema.items():
            if 'type' in field_metadata and field_metadata['type'] in {'integer', 'long', 'float', 'double'}:
                if self[field].dropna().count() / n > 0.5:
                    return field  # Best choice

        # 2️⃣ Use `_seq_no` if available (for stable ordering)
        if "_seq_no" in schema:
            return "_seq_no"

        # 3️⃣ Use `_id` if nothing else is found
        return "_id" if "_id" in schema else None  # If None, random_score will work without a field

    def get_best_field_to_sort(self):
        """
        Determines the best field for sorting:
        1. Prefers a date field if available.
        2. Uses a dedicated keyword field next.
        3. Falls back to a `.keyword` sub-field if available.
        4. Defaults to `_id` if no suitable field is found.
        """
        schema = self.schema
        n = self.count()

        # 1️⃣ Prioritize date fields first
        for field, field_metadata in schema.items():
            if 'type' in field_metadata and field_metadata['type'] == 'date':
                if self[field].dropna().count() / n > 0.5:
                    return field  # Best option for sorting

        # 2️⃣ Check for dedicated keyword fields
        for field, field_metadata in schema.items():
            if 'type' in field_metadata and field_metadata['type'] == 'keyword':
                if self[field].dropna().count() / n > 0.5:
                    return field

        # 3️⃣ Check for `.keyword` sub-fields in text fields
        for field, field_metadata in schema.items():
            if 'fields' in field_metadata and 'keyword' in field_metadata['fields']:
                f = f"{field}.keyword"
                if self[field].dropna().count() / n > 0.5:
                    return f

        # 4️⃣ Default fallback
        return "_doc" if "_doc" in schema else "_id"

    @cached_property
    def best_field_to_randomize(self):
        return self.get_best_field_to_randomize()

    @cached_property
    def best_field_to_sort(self):
        return self.get_best_field_to_sort()

    def sample(self, n=1, seed=None, as_df=True, field=None):
        if field is None:
            field = self.best_field_to_randomize
        ind = self.random_generator(seed, field=field)
        if as_df:
            return ind.as_df(size=n, use_score=True)
        return ind.as_dict(size=n, add_score=True)

    @property
    def ids(self):
        m = self._get_all_metadata()
        return [x['id'] for x in m]

    def dropna(self, subset=None):
        if subset is None:
            subset = self.fields
        if type(subset) is str:
            subset = [subset]
        # add filter to remove None values
        q = Q('exists', field=subset[0])
        for field in subset[1:]:
            q &= Q('exists', field=field)
        return self & q

    def isna(self, subset=None):
        if subset is None:
            subset = self.fields
        if type(subset) is str:
            subset = [subset]
        # add filter to remove None values
        q = Q('missing', field=subset[0])
        for field in subset[1:]:
            q = q | Q('missing', field=field)
        return self & q

    def drop_duplicates(self, subset=None):
        if subset is None:
            subset = self.fields
        if type(subset) is str:
            subset = [subset]
        return self.groupby(subset).agg('first')

