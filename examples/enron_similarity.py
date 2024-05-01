from functools import partial, cached_property

import pandas as pd
import spacy
import re

from src.beam import resource, Timer, BeamData
from src.beam.transformer import Transformer
from src.beam import beam_logger as logger
from src.beam.algorithm import Algorithm

from src.beam.config import TransformerConfig, BeamParam
from src.beam.similarity import SimilarityConfig, TFIDFConfig

class TicketSimilarityConfig(SimilarityConfig, TFIDFConfig):

    # "en_core_web_trf"

    defaults = {
        'chunksize': 1000,
        'n_workers': 40,
        'mp_method': 'apply_async',
        'store_chunk': True,
        'store_path': None,
        'store_suffix': '.parquet',
        'override': False,

    }
    parameters = [
        BeamParam('nlp-model', type=str, default="en_core_web_sm", help='Spacy NLP model'),
        BeamParam('nlp-max-length', type=int, default=2000000, help='Spacy NLP max length'),
        BeamParam('path-to-data', type=str, default='/home/hackathon_2023/data/enron/emails.parquet',
                  help='Path to emails.parquet data'),
        BeamParam('root-path', type=str, default='/home/shared/data/results/enron/',
                  help='Root path to store results'),
        BeamParam('train-ratio', type=float, default=0.4, help='Train ratio for split_dataset'),
        BeamParam('val-ratio', type=float, default=0.3, help='Validation ratio for split_dataset'),
        BeamParam('gap-days', type=int, default=3, help='Gap of days between subsets for split_dataset'),
        BeamParam('preprocess-body', type=bool, default=False, help='Preprocess body text'),
        BeamParam('preprocess-title', type=bool, default=False, help='Preprocess title text (subject)'),
        BeamParam('split-dataset', type=bool, default=False, help='Split the dataset'),
        BeamParam('build-dataset', type=bool, default=False, help='Build the dataset'),
        BeamParam('tfidf-similarity', type=bool, default=True, help='Analyse similarity with TFIDF'),
        BeamParam('tokenizer', type=str, default="BAAI/bge-base-en-v1.5", help='Tokenizer model'),
        BeamParam('tokenizer-chunksize', type=int, default=10000, help='Chunksize for tokenizer'),
    ]


class TicketSimilarity(Algorithm):

    @cached_property
    def entity_remover(self):
        return Transformer(self.hparams, func=replace_entity_over_series)

    @cached_property
    def root_path(self):
        return resource(self.get_hparam('root-path'))

    @cached_property
    def dataset(self):
        bd = BeamData.from_path(self.root_path.joinpath('dataset'))
        bd.cache()
        return bd

    @cached_property
    def metadata(self):
        df = resource(self.get_hparam('path-to-data')).read(target='pandas')
        return df

    @cached_property
    def nlp_model(self):
        nlp = spacy.load(self.get_hparam('nlp-model'))
        nlp.max_length = self.get_hparam('nlp-max-length')
        return nlp

    def preprocess_body(self):
        self.entity_remover.transform(self.metadata['body'], nlp=self.nlp_model, transform_kwargs={
            'store_path': self.root_path.joinpath('enron_mails_without_entities_body')})

    def preprocess_title(self):
        self.entity_remover.transform(self.metadata['subject'], nlp=self.nlp_model, transform_kwargs={
            'store_path': self.root_path.joinpath('enron_mails_without_entities_title')})

    def split_dataset(self):
        split_dataset(self.metadata, self.root_path.joinpath('split_dataset'),
                      train_ratio=self.get_hparam('train-ratio'),
                      val_ratio=self.get_hparam('val-ratio'),
                      gap_days=self.get_hparam('gap-days'))

    def build_dataset(self):
        build_dataset(self.root_path)

    @cached_property
    def tokenizer(self):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(self.get_hparam('tokenizer'))

    def tokenize(self, x):
        return self.tokenizer(x)['input_ids']

    @cached_property
    def tfidf_sim(self):
        from src.beam.similarity import TFIDF
        sim = TFIDF(preprocessor=self.tokenize, chunksize=self.get_hparam('tokenizer-chunksize'),
                    n_workers=self.get_hparam('n_workers'))
        return sim

    def fit_tfidf(self):
        self.tfidf_sim.add(self.dataset['x_train'].values)


def split_dataset(df, output_path, train_ratio=0.4, val_ratio=0.3, gap_days=3):

    df['datetime'] = df['date'].str.split('(').str[0].str.strip()
    df['datetime'] = pd.to_datetime(df['datetime'], format="%a, %d %b %Y %H:%M:%S %z", errors='coerce', utc=True)
    df = df.sort_values('datetime')
    df_filt = df[~df['datetime'].isna()]
    valid_times = df_filt['datetime'] >= pd.Timestamp('1999-01-01', tz='UTC')
    df_filt = df_filt[valid_times]
    valid_times = df_filt['datetime'] < pd.Timestamp('2003-01-01', tz='UTC')
    df_filt = df_filt[valid_times]
    df = df_filt.reset_index(drop=False)
    df = df.set_index('datetime')
    y, labels = pd.factorize(df['from'])
    df['label'] = y
    # Parameters
    d = gap_days  # gap in days

    # Compute the timestamps for splitting
    total_rows = len(df)
    train_end_row = int(total_rows * train_ratio)
    val_end_row = int(total_rows * (train_ratio + val_ratio))

    # Calculate the split points by datetime index
    train_end_date = df.index[train_end_row]
    val_start_date = train_end_date + pd.Timedelta(days=d)
    val_end_date = df.index[val_end_row]
    test_start_date = val_end_date + pd.Timedelta(days=d)

    # Perform the splits
    train = df[df.index < train_end_date]
    validation = df[(df.index >= val_start_date) & (df.index < val_end_date)]
    test = df[df.index >= test_start_date]

    # Checking the split dates and sizes
    logger.info("Training end date:", train_end_date)
    logger.info("Validation start date:", val_start_date)
    logger.info("Validation end date:", val_end_date)
    logger.info("Test start date:", test_start_date)

    logger.info("Training set size:", len(train))
    logger.info("Validation set size:", len(validation))
    logger.info("Test set size:", len(test))

    # Save the splits
    resource(output_path).mkdir()
    resource(output_path).joinpath('train.parquet').write(train)
    resource(output_path).joinpath('validation.parquet').write(validation)
    resource(output_path).joinpath('test.parquet').write(test)
    resource(output_path).joinpath('labels.pkl').write(labels)
    logger.info(f"Splitting done. Data saved to: {output_path}")


def replace_entities(text, nlp=None):

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


# def replace_entity_over_series(series):
#     return pd.Series([replace_entities(text) for text in series.values], index=series.index)

def replace_entity_over_series(series, nlp=None):

    if nlp is not None:
        func = partial(replace_entities, nlp=nlp)
    else:
        func = replace_entities

    return series.apply(func)


def build_dataset(root_path, body_path=None, title_path=None, output_path=None):
    root_path = resource(root_path)

    splits_input_path = root_path.joinpath('split_dataset')
    train = resource(splits_input_path).joinpath('train.parquet').read()
    validation = resource(splits_input_path).joinpath('validation.parquet').read()
    test = resource(splits_input_path).joinpath('test.parquet').read()

    body_path = body_path or root_path.joinpath('enron_mails_without_entities_body')
    bd = BeamData.from_path(body_path)
    body = bd.stacked_values

    title_path = title_path or root_path.joinpath('enron_mails_without_entities_title')
    bd = BeamData.from_path(title_path)
    title = bd.stacked_values

    full_text = pd.concat([body, title], axis=1)
    data = full_text.apply(lambda x: f"Title: {x['subject']}\n\nBody:\n{x['body']}", axis=1)
    x_train = data.loc[train['index']].values.tolist()
    x_val = data.loc[validation['index']].values.tolist()
    x_test = data.loc[test['index']].values.tolist()

    y_train = train['label'].values
    y_val = validation['label'].values
    y_test = test['label'].values

    output_path = output_path or root_path.joinpath('dataset')
    bd = BeamData(data={'x_train': x_train, 'x_val': x_val, 'x_test': x_test,
                  'y_train': y_train, 'y_val': y_val, 'y_test': y_test},
                  path=output_path, override=True)
    bd.store()

    logger.info(f"Dataset built and saved to: {output_path}")


def main():
    # from src.beam import BeamData
    # bd = BeamData.from_path('/home/shared/data/results/enron/enron_mails_without_entities', read_metadata=False)
    # bd.cache()

    hparams = TicketSimilarityConfig()
    alg = TicketSimilarity(hparams=hparams)

    if hparams.get('preprocess-body'):
        with Timer(name='transform: replace_entity_over_series in body', logger=logger) as t:
            alg.preprocess_body()

    if hparams.get('preprocess-title'):
        with Timer(name='transform: replace_entity_over_series in title', logger=logger) as t:
            alg.preprocess_title()

    if hparams.get('split-dataset'):
        with Timer(name='split_dataset', logger=logger) as t:
            alg.split_dataset()

    if hparams.get('build-dataset'):
        with Timer(name='build_dataset', logger=logger) as t:
            alg.build_dataset()

    if hparams.get('tfidf-similarity'):
        with Timer(name='fit_tfidf', logger=logger) as t:
            alg.fit_tfidf()

    logger.info('done enron_similarity example')


if __name__ == '__main__':
    main()
