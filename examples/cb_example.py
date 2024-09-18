import numpy as np
import pandas as pd

from beam import logger
from beam import resource
from beam.algorithm.config import CatboostExperimentConfig
from beam.algorithm import CBAlgorithm
from beam import Experiment
from beam.features.feature import InverseOneHotFeature, ScalingFeature, FeaturesAggregator
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

    features_aggregator = FeaturesAggregator(ScalingFeature('numerical', input_columns=range(10),
                                                                    output_columns=columns[:10], add_name_prefix=False),
                                             InverseOneHotFeature('Wilderness_Area', input_columns=range(10, 14)),
                                             InverseOneHotFeature('Soil_Type', input_columns=slice(14, None)),
                                             state_path='/tmp/covtype_features_state',
                                             artifact_path='/tmp/covtype_features_artifact')

    x = features_aggregator.fit_transform(x)
    return {'x': x, 'y': data['y'], 'columns': x.columns}


def main():

    config = CatboostExperimentConfig(loss_function='MultiClass', iterations=100, grow_policy='Depthwise',
                                      boosting_type='Ordered', device='cuda')
    experiment = Experiment(config)

    data = preprocess_covtype()

    dataset = TabularDataset(x=data['x'], y=data['y'], cat_features=['Wilderness_Area', 'Soil_Type'])

    alg = CBAlgorithm(config, experiment=experiment)
    report = alg.fit(dataset)
    logger.info(f"Model trained with {alg.model}")
    print(report)


if __name__ == '__main__':
    main()


