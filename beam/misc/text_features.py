
import pandas as pd
from typing import List, Union
import textstat


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