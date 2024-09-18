import re

from ..type import BeamType, Types


def svd_preprocess(x):

    x_type = BeamType.check_minor(x)

    if x_type.minor == Types.tensor:
        crow_indices = x.crow_indices().numpy()
        col_indices = x.col_indices().numpy()
        values = x.values().numpy()

        # Create a SciPy CSR matrix
        from scipy.sparse import csr_matrix
        x = csr_matrix((values, col_indices, crow_indices), shape=x.size())
    return x


def replace_entities(text, nlp=None):

    import spacy

    if nlp is None:
        nlp = spacy.load("en_core_web_sm")

    text = text[:nlp.max_length]

    intra_email_regex = r'\b[A-Za-z0-9./\-_]+@[A-Za-z]+\b'
    text = re.sub(intra_email_regex, '<INTRA EMAIL>', text)

    # Use a regular expression to find and replace email addresses
    email_regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    text = re.sub(email_regex, '<EMAIL>', text)

    # filter out strings that end with /ECT and replace by <OTHER_EMAIL>
    text = re.sub(r'\b[A-Za-z0-9./\-_]+/ECT\b', '<OTHER_EMAIL>', text)

    # filter out uri (universal resource identifier) and replace by <URI>
    # Regex pattern to match URIs with various schemes
    pattern = r'\b(\w+):\/\/[^\s,]+'

    # Function to use for replacing each match
    def replace_with_scheme(match):
        scheme = match.group(1)  # Capture the scheme part of the URI
        return f'<{scheme.upper()}>'

    # Replace URIs in the text with their respective scheme tokens
    text = re.sub(pattern, replace_with_scheme, text)

    text = text[:nlp.max_length]

    doc = nlp(text)
    sorted_entities = sorted(doc.ents, key=lambda ent: ent.start_char)

    last_idx = 0
    new_text = []

    for ent in sorted_entities:
        # Append text from last index to start of the entity
        new_text.append(text[last_idx:ent.start_char])

        # Append the appropriate placeholder
        if ent.label_ in ["PERSON", "DATE", "TIME"]:
            placeholder = f"<{ent.label_.lower()}>"
            new_text.append(placeholder)
        else:
            # If not an entity of interest, append the original text
            new_text.append(text[ent.start_char:ent.end_char])

        # Update last index to end of the entity
        last_idx = ent.end_char

    # Append any remaining text after the last entity
    new_text.append(text[last_idx:])

    text = ''.join(new_text)

    # replace with re any sequence of digits with <NUMBER>
    text = re.sub(r'\d+', '<NUMBER>', text)

    return text