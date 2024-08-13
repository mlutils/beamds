from pulearn import BaggingPuClassifier
from sklearn.svm import SVC


def main():

    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=10, n_clusters=2,
                               weights=[0.1, 0.9], flip_y=0, random_state=1)

    svc = SVC(C=10, kernel='rbf', gamma=0.4, probability=True)
    pu_estimator = BaggingPuClassifier(estimator=svc, n_estimators=15)
    pu_estimator.fit(X, y)


if __name__ == '__main__':
    main()
