import numpy as np

from src.beam import Experiment, BeamHparams
from examples.mnist_example import MNISTDataset, MNISTAlgorithm


class MNISTHparams(BeamHparams):
    defaults = dict(project_name='mnist', algorithm='MNISTAlgorithm', amp=False, accelerate=True,
                    device=0, n_epochs=10, epoch_length=1000, objective='acc', model_dtype='float16',
                    stop_at=.99, scheduler='exponential', gamma=.999, scale_epoch_by_batch_size=False)


if __name__ == '__main__':

    # in this example we do not set the root-dir and the path-to-data, so the defaults will be used

    hparams = MNISTHparams()
    experiment = Experiment(hparams)

    dataset = MNISTDataset(hparams)
    alg = MNISTAlgorithm(hparams)

    # train
    alg = experiment.fit(Alg=alg, Dataset=dataset)

    examples = alg.dataset[np.random.choice(len(alg.dataset), size=50000, replace=True)]
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

