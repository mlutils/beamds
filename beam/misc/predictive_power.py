from ..utils import tqdm_beam as tqdm
from sklearn.tree import DecisionTreeClassifier
import pandas as pd


def predictive_power_score(x_train, x_test, y_train, y_test, objective_func=None, depths=None, dt_kwargs=None):

    if dt_kwargs is None:
        dt_kwargs = {}

    if depths is None:
        depths = [1, 2, 4, 6, 10]

    if objective_func is None:
        from sklearn.metrics import accuracy_score
        objective_func = accuracy_score

    objective = {}
    for i in tqdm(x_train.columns):
        xi_train = x_train[[i]]
        xi_test = x_test[[i]]
        oi = {}
        for n in depths:
            alg = DecisionTreeClassifier(max_depth=n, **dt_kwargs)
            alg.fit(xi_train, y_train)
            pred = alg.predict(xi_test)
            oi[n] = objective_func(y_test, pred)
        objective[i] = oi

    df = pd.DataFrame.from_dict(objective, orient='index')
    return df.max(axis=1).sort_values(ascending=False)
