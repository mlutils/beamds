import numpy as np
import pandas as pd

from src.beam import Timer, resource
from src.beam import beam_logger as logger
import os

from examples.enron_similarity.alg import EnronTicketSimilarity
from examples.enron_similarity.config import TextGroupExpansionConfig


def run_enron():
    # from src.beam import BeamData
    # bd = BeamData.from_path('/home/shared/data/results/enron/enron_mails_without_entities', read_metadata=False)
    # bd.cache()

    script_dir = os.path.dirname(os.path.realpath(__file__))
    conf_path = resource(os.path.join(script_dir, 'conf.json')).str
    hparams = TextGroupExpansionConfig(conf_path)
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
            alg.fit_tfidf(subset='validation')
            alg.fit_tfidf(subset='test')

    if hparams.get('calc-dense'):
        with Timer(name='fit_dense', logger=logger) as t:
            alg.fit_dense(subset='validation')
            alg.fit_dense(subset='test')

    if hparams.get('build-features'):
        with Timer(name="Building dataset features for the classifier", logger=logger):
            alg.build_features()

    if hparams.get('save-state') and hparams.get('model-state-path') is not None:
        logger.info(f"Saving state to {hparams.get('model-state-path')}")
        alg.save_state(hparams.get('model-state-path'), override=False)

    # alg.tfidf_sim.metric = 'bm25'
    query = "I need to know about the project"
    results = alg.search_tfidf(query, k=5)

    logger.info(f"TFIDF Results for query: {query}")
    logger.info(results)

    for i in results.index[0]:
        logger.info(alg.subsets['train'].loc[i].values['body'].values[0])

    results = alg.search_dense([query], k=5)
    logger.info(f"Dense Results for query: {query}")
    logger.info(results)

    for i in results.index[0]:
        logger.info(alg.subsets['train'].loc[i].values['body'].values[0])
    logger.info('done enron_similarity example')

    alg.fitted_subset_dense = 'validation'
    alg.fitted_subset_tfidf = 'validation'
    alg.tfidf_sim.metric = 'bm25'
    pd.Series(alg.y['validation']).value_counts().head(20)
    l = 36
    res = alg.build_group_dataset(l, k_sparse=50, k_dense=50)
    tp = (res['y_unlabeled_true'] == l).sum()
    fp = len(res['y_unlabeled_true']) - tp
    p = (alg.y['validation'] == l).sum()
    recall = tp / p
    precision = tp / (tp + fp)
    prevalence = p / len(alg.y['validation'])
    print(f"recall: {recall}, precision: {precision}, prevalence: {prevalence}")

    vu = res['x_unlabeled']
    vp = res['x_pos']
    x = vu + vp
    v = alg.build_features(x)
    y = np.concatenate([res['y_unlabeled'], res['y_pos']])
    with Timer():
        from pulearn import BaggingPuClassifier
        from sklearn.svm import SVC
        svc = SVC(C=10, kernel='rbf', gamma=0.4, probability=True)
        pu_estimator = BaggingPuClassifier(estimator=svc, n_estimators=15, n_jobs=40)
        pu_estimator.fit(v, y)

    # dataset = alg.build_train_test_datasets(l, k_sparse=10, k_dense=10, test_subset='test')
    # from src.beam.config import CatboostConfig
    # from src.beam.algorithm import CBAlgorithm
    # conf = CatboostConfig(cb_depth=8, cb_n_estimators=500, cb_log_resolution=50)
    # cb = CBAlgorithm(conf)
    # from sklearn.model_selection import train_test_split
    # x_train, x_val, y_train, y_val = train_test_split(dataset['x_train'], dataset['y_train'])
    # cb.fit(x_train, y_train, eval_set=(x_val, y_val), beam_postprocess=False)
    # pred = cb.predict(dataset['x_test'])
    # from sklearn.metrics import precision_recall_fscore_support
    # precision_recall_fscore_support(dataset['y_test'], pred)


if __name__ == '__main__':
    run_enron()
