from beam import resource
from sklearn.ensemble import GradientBoostingClassifier
from pulearn import BaggingPuClassifier
from argparse import Namespace
import optuna
import time
from beam import logger


def f_beta(r, p, beta=1, eps=1e-6):
    return (1 + beta**2) * r * p / (beta**2 * p + r + eps)


def optimize_group(alg, label, precision_threshold=.1, beta=2, name=None, storage=None, load_if_exists=True,
                   n_jobs=1, n_trials=100, seed=0):

    t = time.strftime("%Y%m%d-%H%M%S")
    if name is None:
        name = ''
    else:
        name = f"{name}-"

    name = f'enron-sim-{name}label-{label}-{t}'
    if storage is None:
        storage = 'sqlite:///ticket-sim.db'

    study = optuna.create_study(direction='maximize', storage=storage, study_name=name, load_if_exists=load_if_exists)

    def objective(trial):

        gradient_boosting_params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),
            'subsample': trial.suggest_float('subsample', 0.5, 1),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1, log=True),
        }

        pu_params = {
            'n_estimators': trial.suggest_int('n_estimators', 10, 50),
            'max_features': trial.suggest_float('max_features', 0.7, 1.),
            'max_samples': trial.suggest_float('max_samples', 0.7, 1.),
            'oob_score': trial.suggest_categorical('oob_score', [False, True]),
        }

        k_sparse = trial.suggest_int('k_sparse', 5, 20)
        k_dense = trial.suggest_int('k_dense', 5, 20)
        threshold = trial.suggest_float('threshold', 0.4, 0.95)

        base_classifier = GradientBoostingClassifier(random_state=seed, **gradient_boosting_params)
        pu_classifier = BaggingPuClassifier(estimator=base_classifier, verbose=10, random_state=seed, **pu_params)

        res = alg.evaluate(label, k_sparse=k_sparse, k_dense=k_dense, threshold=threshold, pu_classifier=pu_classifier)
        try:
            s = res.metrics
            score = (s.final_precision > precision_threshold) * f_beta(s.final_recall, s.final_precision, beta=beta)
            logger.info(f"Label: {label}, Score: {score}")

            for k, v in vars(s).items():
                v = v.tolist() if hasattr(v, 'tolist') else v
                trial.set_user_attr(k, v)

        except Exception as e:
            logger.error(f"Error: {e}")
            score = 0.0

        return score

    logger.critical(f"Starting hyperparameter optimization (label {label}): {name}")
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)

    logger.critical(f"Finished hyperparameter optimization (label {label}): {name}")
    logger.critical(f"Best score: {study.best_value}")
    logger.critical(f"Best parameters: {study.best_params}")
    return study



