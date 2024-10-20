from functools import partial
from typing import Union, List

import pandas as pd
import spacy
import re

from beam import resource, Timer, BeamData
from beam import beam_logger as logger
import textstat


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

    train = train.reset_index().set_index('message_id').sort_index()
    validation = validation.reset_index().set_index('message_id').sort_index()
    test = test.reset_index().set_index('message_id').sort_index()

    # Checking the split dates and sizes
    logger.info(f"Training end date: {train_end_date}")
    logger.info(f"Validation start date: {val_start_date}")
    logger.info(f"Validation end date: {val_end_date}")
    logger.info(f"Test start date: {test_start_date}")

    logger.info(f"Training set size: {len(train)}")
    logger.info(f"Validation set size: {len(validation)}")
    logger.info(f"Test set size: {len(test)}")

    # Save the splits
    bd = BeamData(data={'train': train, 'validation': validation, 'test': test, 'labels': labels},
                  path=output_path, override=True)
    bd.store()
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


def build_dataset(subsets, root_path, body_path=None, title_path=None, output_path=None):
    root_path = resource(root_path)

    train = subsets['train'].values
    validation = subsets['validation'].values
    test = subsets['test'].values

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


def _extract_textstat_features(text):
    features = {
        'flesch_reading_ease': textstat.flesch_reading_ease(text),
        'smog_index': textstat.smog_index(text),
        'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
        'coleman_liau_index': textstat.coleman_liau_index(text),
        'automated_readability_index': textstat.automated_readability_index(text),
        'dale_chall_readability_score': textstat.dale_chall_readability_score(text),
        'difficult_words': textstat.difficult_words(text),
        'linsear_write_formula': textstat.linsear_write_formula(text),
        'gunning_fog': textstat.gunning_fog(text),
        'text_standard': textstat.text_standard(text, float_output=True),
        'syllable_count': textstat.syllable_count(text),
        'lexicon_count': textstat.lexicon_count(text, removepunct=True),
        'sentence_count': textstat.sentence_count(text),
        'char_count': textstat.char_count(text)
    }
    return features


def extract_textstat_features(text: Union[str, List[str]], n_workers=1) -> pd.DataFrame:
    if isinstance(text, str):
        text = [text]

    if n_workers > 1:
        from pandarallel import pandarallel
        pandarallel.initialize(progress_bar=False, nb_workers=n_workers)
        df = pd.Series(text).parallel_apply(extract_textstat_features).to_list()
        features = pd.concat(df).reset_index(drop=True)

    else:
        features = pd.DataFrame([_extract_textstat_features(t) for t in text])

    return features


class InvMap:
    def __init__(self, invmap):
        self._invmap = invmap

    def __getitem__(self, ind):
        return self._invmap[ind].values


class Tokenizer:

    def __init__(self, hparams):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(hparams.get('tokenizer'))

    def tokenize(self, x):
        return self.tokenizer(x)['input_ids']
