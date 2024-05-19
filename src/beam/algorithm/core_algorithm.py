from ..processor import Processor
from ..logger import beam_logger as logger


class Algorithm(Processor):
    def __init__(self, hparams, name=None, experiment=None, **kwargs):
        super().__init__(hparams=hparams, name=name, **kwargs)
        self._experiment = None
        self.clear_experiment_properties()
        if experiment is not None:
            self.experiment = experiment

    @property
    def experiment(self):
        logger.debug(f"Fetching the experiment which is currently associated with the algorithm")
        return self._experiment

    # a setter function
    @experiment.setter
    def experiment(self, experiment):
        logger.debug(f"The algorithm is now linked to an experiment directory: {experiment.experiment_dir}")
        self.trial = experiment.trial
        self.hparams = experiment.hparams
        self.clear_experiment_properties()
        self._experiment = experiment

    def clear_experiment_properties(self):

        self.clear_cache('device', 'distributed_training', 'distributed_training_framework', 'hpo', 'rank', 'world_size', 'enable_tqdm', 'n_epochs',
                            'batch_size_train', 'batch_size_eval', 'pin_memory', 'autocast_device', 'model_dtype', 'amp',
                            'scaler', 'swa_epochs')

    def preprocess_inference(self, *args, **kwargs):
        pass

    def postprocess_inference(self, *args, **kwargs):
        pass

    def postprocess_epoch(self, *args, **kwargs):
        pass

    def preprocess_epoch(self, *args, **kwargs):
        pass

    def _fit(self, *args, **kwargs):
        raise NotImplementedError("please implement _fit method in your sub-class")

    def fit(self, *args, **kwargs):

        if self._experiment is None:
            from ..config import ExperimentConfig
            from ..experiment import Experiment
            conf = ExperimentConfig(self.hparams)
            experiment = Experiment(conf)
            self.experiment = experiment

        return self._fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        raise NotImplementedError('predict method not implemented')

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError('evaluate method not implemented')

    def report_scalar(self, name, val, subset=None, aggregation=None, append=None, **kwargs):
        self.reporter.report_scalar(name, val, subset=subset, aggregation=aggregation, append=append, **kwargs)

    def report_data(self, name, val, subset=None, data_type=None, **kwargs):

        if '/' in name:
            dt, name = name.split('/')

            if data_type is None:
                data_type = dt
            else:
                data_type = f"{dt}_{data_type}"

        self.reporter.report_data(name, val, subset=subset, data_type=data_type, **kwargs)

    def report_image(self, name, val, subset=None, **kwargs):
        self.reporter.report_image(name, val, subset=subset, **kwargs)

    def report_images(self, name, val, subset=None, **kwargs):
        self.reporter.report_images(name, val, subset=subset, **kwargs)

    def report_scalars(self, name, val, subset=None, **kwargs):
        self.reporter.report_scalars(name, val, subset=subset, **kwargs)

    def report_histogram(self, name, val, subset=None, **kwargs):
        self.reporter.report_histogram(name, val, subset=subset, **kwargs)

    def report_figure(self, name, val, subset=None, **kwargs):
        self.reporter.report_figure(name, val, subset=subset, **kwargs)

    def report_video(self, name, val, subset=None, **kwargs):
        self.reporter.report_video(name, val, subset=subset, **kwargs)

    def report_audio(self, name, val, subset=None, **kwargs):
        self.reporter.report_audio(name, val, subset=subset, **kwargs)

    def report_embedding(self, name, val, subset=None, **kwargs):
        self.reporter.report_embedding(name, val, subset=subset, **kwargs)

    def report_text(self, name, val, subset=None, **kwargs):
        self.reporter.report_text(name, val, subset=subset, **kwargs)

    def report_mesh(self, name, val, subset=None, **kwargs):
        self.reporter.report_mesh(name, val, subset=subset, **kwargs)

    def report_pr_curve(self, name, val, subset=None, **kwargs):
        self.reporter.report_pr_curve(name, val, subset=subset, **kwargs)

    def get_scalar(self, name, subset=None, aggregate=False):
        v = self.reporter.get_scalar(name, subset=subset, aggregate=aggregate)
        return self.reporter.stack_scalar(v)

    def get_scalars(self, name, subset=None, aggregate=False):
        d = self.reporter.get_scalars(name, subset=subset, aggregate=aggregate)
        for k, v in d.items():
            d[k] = self.reporter.stack_scalar(v)
        return d

    def get_data(self, name, subset=None, data_type=None):

        if '/' in name:
            dt, name = name.split('/')

            if data_type is None:
                data_type = dt
            else:
                data_type = f"{dt}_{data_type}"

        return self.reporter.get_data(name, subset=subset, data_type=data_type)

    def get_image(self, name, subset=None):
        return self.reporter.get_image(name, subset=subset)

    def get_images(self, name, subset=None):
        return self.reporter.get_images(name, subset=subset)

    def get_histogram(self, name, subset=None):
        return self.reporter.get_histogram(name, subset=subset)

    def get_figure(self, name, subset=None):
        return self.reporter.get_figure(name, subset=subset)

    def get_video(self, name, subset=None):
        return self.reporter.get_video(name, subset=subset)

    def get_audio(self, name, subset=None):
        return self.reporter.get_audio(name, subset=subset)

    def get_embedding(self, name, subset=None):
        return self.reporter.get_embedding(name, subset=subset)

    def get_text(self, name, subset=None):
        return self.reporter.get_text(name, subset=subset)

    def get_mesh(self, name, subset=None):
        return self.reporter.get_mesh(name, subset=subset)

    def get_pr_curve(self, name, subset=None):
        return self.reporter.get_pr_curve(name, subset=subset)
