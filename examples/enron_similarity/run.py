import numpy as np

from src.beam import Timer, resource
from src.beam import beam_logger as logger
import os

from examples.enron_similarity.alg import TicketSimilarity
from examples.enron_similarity.config import TicketSimilarityConfig


def run_enron():
    # from src.beam import BeamData
    # bd = BeamData.from_path('/home/shared/data/results/enron/enron_mails_without_entities', read_metadata=False)
    # bd.cache()

    script_dir = os.path.dirname(os.path.realpath(__file__))
    conf_path = resource(os.path.join(script_dir, 'conf.json')).str
    hparams = TicketSimilarityConfig(conf_path)
    alg = TicketSimilarity(hparams=hparams)
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
            alg.fit_tfidf()

    if hparams.get('calc-dense'):
        with Timer(name='fit_dense', logger=logger) as t:
            alg.fit_dense()

    if hparams.get('save-state') and hparams.get('model-state-path') is not None:
        logger.info(f"Saving state to {hparams.get('model-state-path')}")
        alg.save_state(hparams.get('model-state-path'))

    # alg.tfidf_sim.metric = 'bm25'
    # query = "I need to know about the project"
    # results = alg.search_tfidf(query, k=5)
    #
    # logger.info(f"TFIDF Results for query: {query}")
    # logger.info(results)
    #
    # for i in results.index[0]:
    #     logger.info(alg.subsets['train'].loc[i].values['body'].values[0])
    #
    # results = alg.search_dense([query], k=5)
    # logger.info(f"Dense Results for query: {query}")
    # logger.info(results)
    #
    # for i in results.index[0]:
    #     logger.info(alg.subsets['train'].loc[i].values['body'].values[0])
    # logger.info('done enron_similarity example')

    l = 1930
    k_sparse = 50
    k_dense = 50

    ind_l = np.where(alg.dataset['y_train'].values == l)[0]

    v = alg.dataset['x_train'].values
    x_l = [v[i] for i in ind_l]

    alg.tfidf_sim.metric = 'bm25'
    results = alg.search_tfidf(x_l[:2], k=k_sparse)

    logger.info(f"TFIDF Results for query: {l}")
    logger.info(results)

    for i in results.index[0]:
        logger.info(alg.subsets['train'].loc[i].values['body'].values[0])

if __name__ == '__main__':
    run_enron()
