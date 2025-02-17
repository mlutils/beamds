
from datetime import datetime
import re

def convert_datetimes(data):
    if isinstance(data, dict):
        return {k: convert_datetimes(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_datetimes(item) for item in data]
    elif isinstance(data, datetime):
        return data.isoformat()
    else:
        return data


def ensure_rfc1123_compliance(name):
    """
    Ensures that a name complies with RFC 1123 by:
    - Lowercase all characters.
    - Replacing any invalid characters with dashes.
    - Replacing consecutive dashes with a single dash.
    - Trimming to the maximum allowed length for Kubernetes names (63 characters).
    - Ensuring the name doesn't start or end with a dash.
    - Handling empty strings by assigning a default name.
    """
    # Lowercase the name and replace invalid characters
    name = name.lower()
    name = re.sub(r"[^a-z0-9-]", "-", name)

    # Replace consecutive dashes with a single dash
    name = re.sub(r"-{2,}", "-", name)

    # Remove leading and trailing dashes
    name = name.strip('-')

    # Trim to the maximum length of 63 characters
    name = name[:63]

    # Remove leading and trailing hyphens again in case they were introduced by truncation
    name = name.strip('-')

    # Handle the case where the name might be empty after stripping
    if not name:
        name = 'default-name'
    # todo: use generator + deployment suffix to generate unique names not hardcoding

    return name