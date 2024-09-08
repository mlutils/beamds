import numpy as np
import pandas as pd

from beam import logger
from beam import resource
from beam.algorithm.config import CatboostExperimentConfig
from beam.algorithm import CBAlgorithm
from beam import Experiment
from beam.features.feature import InverseOneHotFeature, ScalingFeature
from beam.dataset.tabular_dataset import TabularDataset


def get_data(name='covtype'):
    from sklearn import datasets
    fetcher = getattr(datasets, f'fetch_{name}')
    data = fetcher()
    return {'x': data.data, 'y': data.target, 'columns': data.feature_names, 'labels': data.target}


def preprocess_covtype():
    data = get_data(name='covtype')
    x = data['x']
    columns = data['columns']

    x1 = ScalingFeature(columns=columns[:10]).fit_transform(x[:, :10])
    x2 = InverseOneHotFeature(name='Wilderness_Area').transform(x[:, 10:14])
    x3 = InverseOneHotFeature(name='Soil_Type').transform(x[:, 14:])

    x = pd.concat([x1, x2, x3], axis=1)
    return {'x': x, 'y': data['y'], 'columns': x.columns}


def main():

    config = CatboostExperimentConfig(loss_function='MultiClass', iterations=100)
    experiment = Experiment(config)

    data = preprocess_covtype()

    dataset = TabularDataset(x=data['x'], y=data['y'], cat_features=['Wilderness_Area', 'Soil_Type'])

    alg = CBAlgorithm(config, experiment=experiment)
    report = alg.fit(dataset)
    logger.info(f"Model trained with {alg.model}")
    print(report)


if __name__ == '__main__':
    main()


