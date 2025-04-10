from elasticsearch_dsl import Q, Document, Text, Keyword, Integer, Float, Boolean, Date, DenseVector
import re
import pandas as pd
from io import StringIO

def parse_kql_to_dsl(kql):
    """
    Parse a KQL string into an elasticsearch_dsl.Q object.
    Supports:
    - Basic field:value pairs
    - Logical operators: AND, OR, NOT
    - Numerical ranges (>, <, >=, <=)
    - Boolean values
    - Wildcards (*, ?)
    - Exact matches (using double quotes around values)
    """

    def translate_logical_op(op):
        """Map logical operators to Elasticsearch bool components."""
        op = op.upper()
        if op == "AND":
            return "must"
        elif op == "OR":
            return "should"
        elif op == "NOT":
            return "must_not"
        return None

    def handle_value(field, value):
        """Parse value for wildcards, ranges, or types."""
        if "*" in value or "?" in value:
            # Wildcard handling
            return Q("wildcard", **{field: value})
        elif re.match(r"^[<>]=?\d+(\.\d+)?$", value):
            # Numerical range
            match = re.match(r"([<>]=?)(\d+(\.\d+)?)", value)
            operator, num_value = match.groups()[0], float(match.groups()[1])
            range_key = {"<": "lt", "<=": "lte", ">": "gt", ">=": "gte"}[operator]
            return Q("range", **{field: {range_key: num_value}})
        elif value.lower() in {"true", "false"}:
            # Boolean value
            return Q("term", **{field: value.lower() == "true"})
        elif value.startswith('"') and value.endswith('"'):
            # Exact match (term query)
            exact_value = value.strip('"')
            return Q("term", **{field: exact_value})
        else:
            # Default to match
            return Q("match", **{field: value})

    tokens = kql.split()
    stack = []
    bool_query = {"must": [], "should": [], "must_not": []}

    i = 0
    while i < len(tokens):
        token = tokens[i]

        if token in {"AND", "OR", "NOT", "and", "or", "not"}:
            operator = translate_logical_op(token)
            stack.append(operator)
        elif ":" in token:
            field, value = token.split(":", 1)
            if not len(value):
                value = tokens[i + 1]
                i += 1
            query_part = handle_value(field, value)

            # Attach to bool query
            if stack and stack[-1] in {"must", "should", "must_not"}:
                bool_query[stack.pop()].append(query_part)
            else:
                bool_query["must"].append(query_part)
        elif token.startswith("("):  # Start of a grouped expression
            stack.append("(")
        elif token.endswith(")"):  # End of a grouped expression
            # TODO: Implement grouping logic
            stack.pop()
        i += 1

    dsl_query = Q("bool", **{k: v for k, v in bool_query.items() if v})
    return dsl_query


# Mapping Elasticsearch types to elasticsearch-dsl field types
FIELD_TYPE_MAPPING = {
    "text": Text,
    "keyword": Keyword,
    "integer": Integer,
    "float": Float,
    "boolean": Boolean,
    "date": Date,
    "dense_vector": DenseVector,  # For vector fields
}


def generate_document_class(client, index_name):
    # Retrieve the index mapping
    mapping = client.indices.get_mapping(index=index_name)
    properties = mapping[index_name]["mappings"]["properties"]

    # Create fields dynamically
    fields = {}
    for field_name, field_info in properties.items():
        field_type = field_info.get("type")
        if field_type in FIELD_TYPE_MAPPING:
            if field_type == "dense_vector":
                dims = field_info.get("dims", 0)  # Retrieve dimensions for DenseVector
                fields[field_name] = FIELD_TYPE_MAPPING[field_type](dims=dims)
            else:
                fields[field_name] = FIELD_TYPE_MAPPING[field_type]()

    # Dynamically define the Document class
    class DynamicDocument(Document):
        class Index:
            name = index_name

    # Add the fields to the Document class
    for field_name, field in fields.items():
        setattr(DynamicDocument, field_name, field)

    return DynamicDocument


def describe_dataframe(df: pd.DataFrame, n_samples: int = 5, include_info: bool = True,
                       include_stats: bool = True, include_sample: bool = True, include_corr: bool = False) -> str:
    """
    Returns a string summary of a DataFrame, including metadata, statistics, and sample rows.

    Parameters:
    - df (pd.DataFrame): The DataFrame to describe.
    - n_samples (int): Number of sample rows to include.
    - include_info (bool): Whether to include df.info() output.
    - include_stats (bool): Whether to include statistical summary.
    - include_sample (bool): Whether to include sample rows.

    Returns:
    - str: Formatted string containing the DataFrame description.
    """
    output = StringIO()

    summary_parts = []

    if include_info:
        df.info(buf=output)
        summary_parts.append("\n**DataFrame Info:**\n" + output.getvalue())
        output.seek(0)
        output.truncate(0)

    summary_parts.append(f"\n**Shape:** {df.shape}")
    summary_parts.append(f"\n**Columns:** {list(df.columns)}")
    summary_parts.append(f"\n**Data Types:**\n{df.dtypes}")
    summary_parts.append(f"\n**Memory Usage:**\n{df.memory_usage(deep=True)}")

    if include_stats:
        summary_parts.append("\n**Descriptive Statistics:**\n" + str(df.describe(include='all')))
        summary_parts.append(f"\n**Unique Values per Column:**\n{df.nunique()}\n")
        summary_parts.append(f"\n**Missing Values per Column:**\n{df.isnull().sum()}\n")
        if include_corr and not df.select_dtypes(include=['number']).empty:
            summary_parts.append(f"\n**Correlation Matrix:**\n{df.corr()}\n")

    if include_sample:
        summary_parts.append(f"\n**Sample Rows ({n_samples}):**\n{df.sample(min(n_samples, len(df)))}\n")

    return "\n".join(summary_parts)