from pulearn import BaggingPuClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier


def synthetic_test(estimator='svc'):

    X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=10, n_clusters_per_class=2,
                                   weights=[0.1, 0.9], flip_y=0, random_state=1)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    if estimator == 'svc':
        estimator = SVC(C=10, kernel='rbf', gamma=0.4, probability=True)
    elif estimator == 'catboost':
        estimator = CatBoostClassifier(iterations=100, depth=2, learning_rate=0.1, loss_function='Logloss',
                                       verbose=20)
    else:
        raise ValueError(f"Invalid estimator: {estimator}")

    pu_estimator = BaggingPuClassifier(estimator=estimator, n_estimators=15, verbose=20)
    pu_estimator.fit(x_train, y_train)

    y_pred = pu_estimator.predict(x_test)
    print(precision_recall_fscore_support(y_test, y_pred))


def covtype_test():

    from examples.cb_example import preprocess_covtype

    data = preprocess_covtype()
    x = data['x']
    y = data['y']
    y = (y == 1).astype(int)  # Convert to binary
    categorical_features = ['Wilderness_Area', 'Soil_Type']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    model = CatBoostClassifier(iterations=100, depth=2, learning_rate=0.1, loss_function='Logloss', verbose=20,
                               cat_features=categorical_features)

    pu_estimator = BaggingPuClassifier(estimator=model, n_estimators=15, verbose=20)
    pu_estimator.fit(x_train, y_train)

    y_pred = pu_estimator.predict(x_test)
    print(precision_recall_fscore_support(y_test, y_pred))


def main():
    # synthetic_test('svc')
    # synthetic_test('catboost')
    covtype_test()


if __name__ == '__main__':
    main()
