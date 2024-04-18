from functools import partial

import pandas as pd
import spacy

from src.beam import resource, Timer
from src.beam.transformer import Transformer
from src.beam import beam_logger as logger

from src.beam.config import TransformerConfig, BeamParam


class TicketSimilarityConfig(TransformerConfig):

    defaults = {
        'chunksize': 100,
        'n_workers': 40,
        'mp_method': 'apply_async',
        'store_chunk': True,
        'store_path': '/home/shared/data/results/enron/enron_mails_without_entities_acc',
        'store_suffix': '.parquet',
        'override': False,

    }
    parameters = [
        BeamParam('nlp-model', type=str, default="en_core_web_trf", help='Spacy NLP model'),
        BeamParam('nlp-max-length', type=int, default=2000000, help='Spacy NLP max length'),
        BeamParam('path-to-data', type=str, default='/home/hackathon_2023/data/enron/emails.parquet',
                  help='Path to emails.parquet data')
    ]


def replace_entities(text, nlp=None):

    if nlp is None:
        nlp = spacy.load("en_core_web_sm")

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

    # from src.beam import BeamData
    # bd = BeamData.from_path('/home/shared/data/results/enron/enron_mails_without_entities', read_metadata=False)
    # bd.cache()

    hparams = TicketSimilarityConfig()

    df = resource(hparams.get('path-to-data')).read(target='pandas')

    nlp = spacy.load(hparams.get('nlp-model'))
    nlp.max_length = hparams.get('nlp-max-length')

    transformer = Transformer(hparams, func=replace_entity_over_series)

    with Timer(name='transform: replace_entity_over_series', logger=logger) as t:
        res = transformer.transform(df['body'], nlp=nlp)

    print('done enron_similarity example')
