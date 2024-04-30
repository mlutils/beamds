from functools import partial

import pandas as pd
import spacy
import re

from src.beam import resource, Timer, BeamData
from src.beam.transformer import Transformer
from src.beam import beam_logger as logger

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
        BeamParam('preprocess-body', type=bool, default=True, help='Preprocess body text'),
        BeamParam('preprocess-subject', type=bool, default=True, help='Preprocess subject text (title)'),
        BeamParam('split-dataset', type=bool, default=False, help='Split the dataset'),
        BeamParam('build-dataset', type=bool, default=True, help='Build the dataset'),
    ]


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
    output_path.mkdir()
    output_path.joinpath('x_train.parquet').write(x_train)
    output_path.joinpath('x_val.parquet').write(x_val)
    output_path.joinpath('x_test.parquet').write(x_test)

    output_path.joinpath('y_train.pkl').write(y_train)
    output_path.joinpath('y_val.pkl').write(y_val)
    output_path.joinpath('y_test.pkl').write(y_test)

    logger.info(f"Dataset built and saved to: {output_path}")


def main():
    # from src.beam import BeamData
    # bd = BeamData.from_path('/home/shared/data/results/enron/enron_mails_without_entities', read_metadata=False)
    # bd.cache()

    hparams = TicketSimilarityConfig()
    root_path = resource(hparams.get('root-path'))

    df = resource(hparams.get('path-to-data')).read(target='pandas')

    nlp = spacy.load(hparams.get('nlp-model'))
    nlp.max_length = hparams.get('nlp-max-length')

    transformer = Transformer(hparams, func=replace_entity_over_series)

    if hparams.get('preprocess-body'):
        with Timer(name='transform: replace_entity_over_series in body', logger=logger) as t:
            transformer.transform(df['body'], nlp=nlp, transform_kwargs={
                                            'store_path': root_path.joinpath('enron_mails_without_entities_body')})

    if hparams.get('preprocess-subject'):
        with Timer(name='transform: replace_entity_over_series in title', logger=logger) as t:
            transformer.transform(df['subject'], nlp=nlp, transform_kwargs={
                                            'store_path': root_path.joinpath('enron_mails_without_entities_title')})

    if hparams.get('split-dataset'):
        split_dataset(df,
                      output_path=root_path.joinpath('split_dataset'),
                      train_ratio=hparams.get('train-ratio'),
                      val_ratio=hparams.get('val-ratio'),
                      gap_days=hparams.get('gap-days'))

    if hparams.get('build-dataset'):
        build_dataset(root_path)

    logger.info('done enron_similarity example')


if __name__ == '__main__':
    main()
