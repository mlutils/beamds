from functools import partial

import pandas as pd
import spacy

from src.beam import resource, Timer
from src.beam.transformer import Transformer
from src.beam import beam_logger as logger


def replace_entities(text, nlp=None):

    if nlp is None:
        nlp = spacy.load("en_core_web_sm")

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

    return ''.join(new_text)


# def replace_entity_over_series(series):
#     return pd.Series([replace_entities(text) for text in series.values], index=series.index)

def replace_entity_over_series(series, nlp=None):

    if nlp is not None:
        func = partial(replace_entities, nlp=nlp)
    else:
        func = replace_entities

    return series.apply(func)


if __name__ == '__main__':

    path = '/home/hackathon_2023/data/enron/emails.parquet'
    df = resource(path).read(target='pandas')
    # Load the English tokenizer, tagger, parser, NER, and word vectors
    nlp = spacy.load("en_core_web_sm")

    transformer = Transformer(func=replace_entity_over_series, chunksize=1000,
                              n_workers=40, mp_method='apply_async', store_chunk=True,
                              store_path='/home/shared/data/results/enron/enron_mails_without_entities',
                              store_suffix='.parquet')

    with Timer(name='transform: replace_entity_over_series', logger=logger) as t:
        res = transformer.transform(df['body'], nlp=nlp)

    print('done enron_similarity example')
