class DSLToKQLParserException(Exception):
    """Custom exception for DSL to KQL parsing errors."""
    pass


def dsl_to_kql(dsl_query, root=True):
    """
    Converts a DSL query (in dictionary format) into a KQL query.

    :param dsl_query: A dictionary representing the DSL query.
    :return: A string representing the equivalent KQL query.
    :raises DSLToKQLParserException: If the DSL query cannot be converted to KQL.
    """
    if not isinstance(dsl_query, dict):
        raise DSLToKQLParserException("The DSL query must be a dictionary.")

    left_wrapper = "" if root else "("
    right_wrapper = "" if root else ")"

    try:
        kql_parts = []

        for key, value in dsl_query.items():
            if key == "match":
                # Handle match queries
                for field, match_value in value.items():
                    kql_parts.append(f"{field}: {match_value}")
            elif key == "term":
                # Handle term queries
                for field, term_value in value.items():
                    kql_parts.append(f"{field}: {term_value}")
            elif key == "range":
                # Handle range queries
                for field, range_values in value.items():
                    range_parts = []
                    for operator, range_value in range_values.items():
                        if operator == "gte":
                            range_parts.append(f">= {range_value}")
                        elif operator == "lte":
                            range_parts.append(f"<= {range_value}")
                        else:
                            raise DSLToKQLParserException(f"Unsupported range operator: {operator}")
                    kql_parts.append(f"{field}: {' and '.join(range_parts)}")
            elif key == "bool":
                # Handle boolean queries (must, should, must_not)
                if "must" in value:
                    must_parts = [dsl_to_kql(sub_query, root=False) for sub_query in value["must"]]
                    kql_parts.append(f"{left_wrapper}{' and '.join(must_parts)}{right_wrapper}")
                if "should" in value:
                    should_parts = [dsl_to_kql(sub_query, root=False) for sub_query in value["should"]]
                    kql_parts.append(f"{left_wrapper}{' or '.join(should_parts)}{right_wrapper}")
                if "must_not" in value:
                    must_not_parts = [dsl_to_kql(sub_query, root=False) for sub_query in value["must_not"]]
                    kql_parts.append(f"not ({' or '.join(must_not_parts)})")
            else:
                raise DSLToKQLParserException(f"Unsupported query type: {key}")

        s =  " and ".join(kql_parts)

        return s

    except Exception as e:
        raise DSLToKQLParserException(f"Failed to parse DSL query: {e}") from e

