import numpy as np
import pandas as pd

from beam import Timer, resource
from beam import beam_logger as logger
import os

from examples.enron_similarity.alg import EnronTicketSimilarity
from examples.enron_similarity.config import TicketSimilarityConfig


def run_enron():

    script_dir = os.path.dirname(os.path.realpath(__file__))
    conf_path = resource(os.path.join(script_dir, 'config.yaml')).str
    hparams = TicketSimilarityConfig(conf_path)
    alg = EnronTicketSimilarity(hparams=hparams)

    print(alg.special_state_attributes)

    if hparams.get('reload-state') and hparams.get('model-state-path') is not None:
        logger.info(f"Loading state from {hparams.get('model-state-path')}")
        alg.load_state(hparams.get('model-state-path'))

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

    if hparams.get('calc-tfidf'):
        with Timer(name='fit_tfidf', logger=logger) as t:
            alg.fit_tfidf(subset='validation')
            alg.fit_tfidf(subset='test')
            alg.fit_tfidf(subset='train')

    if hparams.get('calc-dense'):
        with Timer(name='fit_dense', logger=logger) as t:
            alg.fit_dense(subset='validation')
            alg.fit_dense(subset='test')
            alg.fit_dense(subset='train')

    if hparams.get('build-features'):
        with Timer(name="Building dataset features for the classifier", logger=logger):
            alg.build_features()

    if hparams.get('save-state'):

        override = hparams.get('reload-state')
        logger.info(f"Saving state to {hparams.get('model-state-path')}")
        alg.save_state(hparams.get('model-state-path'), override=override)

    df = alg.search_dual('I need to know about the project', k_sparse=5, k_dense=5)
    print(df)

    # alg.tfidf_sim.metric = 'bm25'
    query = "I need to know about the project"
    results = alg.search_tfidf(query, k=5)

    logger.info(f"TFIDF Results for query: {query}")
    logger.info(results)

    for i in results.index[0]:
        logger.info(alg.subsets['validation'].loc[i].values['body'].values[0])

    results = alg.search_dense([query], k=5)
    logger.info(f"Dense Results for query: {query}")
    logger.info(results)

    for i in results.index[0]:
        logger.info(alg.subsets['validation'].loc[i].values['body'].values[0])
    logger.info('done enron_similarity example')


if __name__ == '__main__':
    run_enron()
