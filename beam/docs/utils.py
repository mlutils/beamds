from elasticsearch_dsl import Q
import re


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

        if token in {"AND", "OR", "NOT"}:
            operator = translate_logical_op(token)
            stack.append(operator)
        elif ":" in token:
            field, value = token.split(":", 1)
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
