from sklearn.datasets import make_classification
import pandas as pd

from beam import UniversalDataset, UniversalConfig, logger, NeuralAlgorithm, Experiment, beam_server
from beam.nn import LinearNet
from torch import nn


class SyntheticClassificationAlgorithm(NeuralAlgorithm):

    def __init__(self, hparams, **kwargs):

        net = LinearNet(hparams.n_features, l_h=hparams.l_h, l_out=hparams.n_classes, n_l=hparams.n_l, bias=hparams.bias,
                        activation=hparams.activation, batch_norm=hparams.batch_norm,
                        input_dropout=hparams.input_dropout, dropout=hparams.dropout)

        super().__init__(hparams, networks=net, **kwargs)
        self.loss = nn.CrossEntropyLoss()


    def train_iteration(self, sample=None, label=None, index=None, counter=None, subset=None, training=True, **kwargs):
        x, y = sample['x'], sample['y']
        net = self.networks['net']
        y_pred = net(x)
        loss = self.loss(y_pred, y)
        self.apply(loss, training=training)
        acc = (y_pred.argmax(dim=1) == y).float().mean()
        self.report_scalar('accuracy', acc)

    def inference_iteration(self, sample=None, label=None, index=None, subset=None, predicting=False, **kwargs):
        if predicting:
            x = sample
        else:
            x, y = sample['x'], sample['y']
        net = self.networks['net']
        y_pred = net(x)

        if not predicting:
            loss = self.loss(y_pred, y)
            self.apply(loss, training=False)
            acc = (y_pred.argmax(dim=1) == y).float().mean()
            self.report_scalar('accuracy', acc)

        return y_pred


if __name__ == '__main__':

    # Generate synthetic dataset

    conf = UniversalConfig(n_samples=40000, n_features=20, n_informative=10, n_redundant=5, n_classes=4,
                           dataset_seed=42, l_h=256, n_l=2, bias=True,
                           activation='ReLU', batch_norm=False, input_dropout=0.0, dropout=0.0,
                           objective='accuracy', n_epochs=2, batch_size=64, lr=0.001)

    X, y = make_classification(n_samples=conf.n_samples,  # Number of samples
                               n_features=conf.n_features,  # Total number of features
                               n_informative=conf.n_informative,  # Number of informative features
                               n_redundant=conf.n_redundant,  # Number of redundant features
                               n_classes=conf.n_classes,  # Number of classes
                               random_state=conf.dataset_seed)  # Reproducibility

    # Convert to DataFrame for better readability
    df = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(X.shape[1])])

    dataset = UniversalDataset(x=df.values, y=y, hparams=conf)
    dataset.split()

    exp = Experiment(conf)
    alg = exp.fit(SyntheticClassificationAlgorithm, dataset=dataset)

    beam_server(alg)


    # to connect and query the model use:
    # alg = resource('http://localhost:<port>')
    # preds = alg.predict(torch.randn(n_examples, n_features))

    logger.warning("Done!")

