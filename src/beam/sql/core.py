from ..processor.core import Processor
from ..data import BeamData
from ..path import beam_key
import pandas as pd
import re
from sqlalchemy.engine import create_engine


class BeamSQL(Processor):

    # Build a beam class that provides an abstraction layer to different databases and lets us develop our tools without committing to a database technology.
    #
    # The class will be based on sqlalchemy+pandas but it can be inherited by subclasses that use 3rd party packages such as pyathena.
    #
    # some key features:
    # 1. the interface will be based on url addresses as in the BeamPath class
    # 2. two levels will be supported, db level where each index is a table and table level where each index is a column.
    # 3. minimizing the use of schemas and inferring the schemas from existing pandas dataframes and much as possible
    # 4. adding pandas like api whenever possible, for example, selecting columns with __getitem__, uploading columns and tables with __setitem__, loc, iloc
    # 5. the use of sqlalchemy and direct raw sql queries will be allowed.

    def __init__(self, *args, llm=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._connection = None
        self._engine = None
        self._table = None
        self._database = None
        self._index = None
        self._columns = None
        self._llm = llm

    @property
    def llm(self):
        return self._llm

    @property
    def database(self):
        return self._database

    @property
    def table(self):
        return self._table

    @property
    def index(self):
        return self._index

    @property
    def columns(self):
        return self._columns

    def set_database(self, database):
        self._database = database

    def set_llm(self, llm):
        self._llm = llm

    def set_index(self, index):
        self._index = index

    def set_columns(self, columns):
        self._columns = columns

    def set_table(self, table):
        self._table = table

    def __getitem__(self, item):

        if not isinstance(item, tuple):
            item = (item,)

        if self.table is None:
            axes = ['table', 'index', 'columns']
        else:
            axes = ['index', 'columns']

        for i, ind_i in enumerate(item):
            a = axes.pop(0)
            if a == 'table':
                self.set_table(ind_i)
            elif a == 'index':
                self.set_index(ind_i)
            elif a == 'columns':
                self.set_columns(ind_i)

        return self

    def sql(self, query, **kwargs):
        return pd.read_sql(query, self._connection, **kwargs)

    def get_sample(self, n=1, **kwargs):
        raise NotImplementedError

    def get_schema(self):
        raise NotImplementedError

    def nlp(self, query, **kwargs):

        schema = self.get_schema()

        prompt = f"Task: generate an SQL query that best describes the following text:\n {query}\n\n" \
                 f"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n" \
                 f"Additional instructions:\n\n" \
                 f"1. The queried table name is: {self.database}.{self.table}\n" \
                 f"2. Assume that the schema for the queried table is:\n{schema}\n\n" \
                 f"3. Here are 4 example rows {self.get_sample(n=4)}\n\n" \
                 f"4. In your response use only valid column names that best match the text\n\n" \
                 f"5. Important: your response must contain only the SQL query and nothing else, and it must be valid.\n\n" \
                 f"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n" \
                 f"Response: \"\"\"\n{{text input here}}\n\"\"\""

        response = self.llm.ask(prompt, **kwargs)

        query = response.choices[0].text
        query = re.sub(r'\"\"\"', '', query)

        return self.sql(query)

    @staticmethod
    def df2table(df, name, metadata=None):

        from sqlalchemy import Table, Column
        from sqlalchemy.schema import MetaData

        if metadata is None:
            metadata = MetaData()

        # Define the SQLAlchemy table object based on the DataFrame
        columns = [column for column in df.columns]
        types = {column: df.dtypes[column].name for column in df.columns}
        table = Table(name, metadata, *(Column(column, types[column]) for column in columns))

        return table

    @property
    def engine(self):
        raise NotImplementedError

    def __enter__(self):
        self._connection = self.engine.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._connection.close()
        self._connection = None


class BeamAthena(BeamSQL):
    def __init__(self, s3_staging_dir, role_session_name=None, region_name=None, access_key=None, secret_key=None,
                 *args, **kwargs):

        self.access_key = beam_key('AWS_ACCESS_KEY_ID', access_key)
        self.secret_key = beam_key('aws_secret_key', secret_key)
        self.s3_staging_dir = s3_staging_dir

        if role_session_name is None:
            role_session_name = "PyAthena-session"
        self.role_session_name = role_session_name

        if region_name is None:
            region_name = "eu-north-1"
        self.region_name = region_name

        state = {'s3_staging_dir': self.s3_staging_dir, 'role_session_name': self.role_session_name,
                      'region_name': self.region_name, 'access_key': self.access_key, 'secret_key': self.secret_key}

        super().__init__(*args, state=state, **kwargs)

    def get_sample(self, n=1, **kwargs):

        from pyathena.pandas.util import as_pandas

        query = f"SELECT * FROM {self.database}.{self.table} LIMIT {n}"

        cursor = self.connection.cursor()
        cursor.execute(query)
        df = as_pandas(cursor)

        return df

    def get_schema(self):

        query = f"DESCRIBE {self.database}.{self.table}"

        cursor = self.connection.cursor()
        cursor.execute(query)
        # Fetch the result
        result = cursor.fetchall()

        return result

    @property
    def engine(self):
        if self._engine is None:
            self._engine = create_engine('athena+pyathena://', creator=lambda: self.connection)
        return self._engine

    @property
    def connection(self):

        if self._connection is None:

            from pyathena import connect

            self._connection = connect(s3_staging_dir=self.s3_staging_dir,
                                       role_session_name=self.role_session_name,
                                       region_name=self.region_name, aws_access_key_id=self.access_key,
                                       aws_secret_access_key=self.secret_key)

        return self._connection

    def sql(self, query):

        from pyathena.pandas.util import as_pandas

        cursor = self.connection.cursor()
        cursor.execute(query)
        df = as_pandas(cursor)
        bd = BeamData(df)

        return bd


# from sqlalchemy import create_engine, func, MetaData, Table, and_
# import pandas as pd
#
#
# class BeamSQL:
#     def __init__(self, uri):
#         self.engine = create_engine(uri)
#         self.metadata = MetaData(bind=self.engine)
#         self.current_table = None
#         self.groupby_columns = []
#         self.filter_conditions = []
#
#     def table(self, table_name):
#         """Select the current working table."""
#         self.current_table = Table(table_name, self.metadata, autoload=True)
#         return self
#
#     def groupby(self, *column_names):
#         self.groupby_columns = column_names
#         return self
#
#     def count(self):
#         if not self.current_table or not self.groupby_columns:
#             raise Exception("Table not selected or columns for grouping not provided")
#
#         columns_to_select = [getattr(self.current_table.c, col) for col in self.groupby_columns]
#         s = select(columns_to_select + [func.count()]) \
#             .where(and_(*self.filter_conditions)) \
#             .group_by(*columns_to_select)
#
#         result = self.engine.execute(s)
#         return pd.DataFrame(result.fetchall(), columns=self.groupby_columns + ['count'])
#
#     def filter(self, column_name, operator, value):
#         column = getattr(self.current_table.c, column_name)
#         op_map = {
#             "==": column.__eq__,
#             ">": column.__gt__,
#             "<": column.__lt__,
#             ">=": column.__ge__,
#             "<=": column.__le__,
#         }
#
#         if operator not in op_map:
#             raise Exception(f"Operator {operator} not supported")
#
#         condition = op_map[operator](value)
#         self.filter_conditions.append(condition)
#         return self
#
#     def query(self, raw_sql):
#         result = self.engine.execute(raw_sql)
#         columns = result.keys()
#         return pd.DataFrame(result.fetchall(), columns=columns)
#
#
# # Sample usage:
# bs = BeamSQL("sqlite:////path/to/sqlite3.db")
#
# # Equivalent to df.groupby(['name', 'age']).count()
# result = bs.table('your_table_name').groupby('name', 'age').count()
# print(result)
#
# # Equivalent to df[df['age'] > 25].groupby(['name']).count()
# result = bs.table('your_table_name').filter('age', '>', 25).groupby('name').count()
# print(result)
#
# # Raw SQL
# result = bs.query("SELECT name, COUNT(*) FROM your_table_name GROUP BY name")
# print(result)
