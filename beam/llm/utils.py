import re
import string
from .model_adapter import get_model_adapter


default_token_pattern = fr" ?[{re.escape(string.whitespace + string.punctuation)}]| ?[A-Za-z]{{1,4}}| ?\d{{1,3}}"


def text_splitter(text, chunk_size=100, separators=["\n\n", ". ", " "], length_function=None):
    if length_function is None:
        length_function = lambda x: int(1.5 * len(re.findall(r'\w+', x)))

    s = separators[0]
    open_chunks = text.split(s)
    closed_chunks = []
    next_chunk = ''

    for c in open_chunks:
        if length_function(c) > chunk_size:
            if len(next_chunk) > 0:
                closed_chunks.append(next_chunk)
            closed_chunks.extend(text_splitter(c, chunk_size, separators[1:], length_function))
            next_chunk = closed_chunks.pop()
        elif length_function(next_chunk) + length_function(c) > chunk_size:
            closed_chunks.append(next_chunk)
            next_chunk = c
        else:
            next_chunk = f"{next_chunk}{s}{c}"

    closed_chunks.append(next_chunk)

    return closed_chunks


def split_to_tokens(s):
    return re.findall(default_token_pattern, s)


def default_tokenizer(text):

    if isinstance(text, list):
        input_ids = [split_to_tokens(t) for t in text]
    else:
        input_ids = split_to_tokens(text)

    return {"input_ids": input_ids}


def estimate_tokens(s):
    # Pattern to capture:
    # 1. Any single punctuation or whitespace character.
    # 2. Sequences of up to 4 alphabetic characters.
    # 3. Sequences of up to 3 numeric characters.

    matches = re.findall(default_token_pattern, s)
    token_count = len(matches)

    # For the remaining unmatched characters in the string (those that aren't part of the recognized patterns),
    # we count each character as a separate token.
    unmatched_characters = re.sub(default_token_pattern, "", s)
    token_count += len(unmatched_characters)

    return token_count


def get_conversation_template(model_path):
    ma = get_model_adapter(model_path)
    return ma.get_default_conv_template('       ')
