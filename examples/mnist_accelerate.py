import numpy as np

from beam import Experiment, BeamConfig
from examples.mnist_example import MNISTDataset, MNISTAlgorithm


class MNISTConfig(BeamConfig):
    defaults = dict(project_name='mnist', algorithm='MNISTAlgorithm', amp=False, training_framework='accelerate',
                    device=1, n_epochs=2, epoch_length=1000, objective='acc', model_dtype='float16',
                    stop_at=.99, scheduler='exponential', gamma=.999, scale_epoch_by_batch_size=False)


if __name__ == '__main__':

    # in this example we do not set the logs-path and the data-path, so the defaults will be used

    hparams = MNISTConfig()
    experiment = Experiment(hparams)

    dataset = MNISTDataset(hparams)
    alg = MNISTAlgorithm(hparams)

    # train
    alg = experiment.fit(alg=alg, dataset=dataset)

    examples = alg.dataset[np.random.choice(len(alg.dataset), size=50000, replace=True)]

    from beam.auto import AutoBeam

    AutoBeam.to_bundle(alg, '/tmp/mnist_bundle')

    res = alg.predict(examples.data['x'])

    # ## Inference
    inference = alg('validation')

    print('Validation inference results:')
    for n, v in inference.statistics['metrics'].items():
        print(f'{n}:')
        print(v)

    inference = alg({'x': examples.data['x'], 'y': examples.data['y']})

    print('Test inference results:')
    for n, v in inference.statistics['metrics'].items():
        print(f'{n}:')
        print(v)

