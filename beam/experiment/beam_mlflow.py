import mlflow.pyfunc
import mlflow
from mlflow import log_params, log_metric
import os
from ..path import beam_path
from ..algorithm import NeuralAlgorithm


class MLflowSummaryWriter:
    def __init__(self,  exp_name, tensorboard_hparams=None, mlflow_uri=None):

        if mlflow_uri is None:
            mlflow_uri = os.environ['MLFLOW_TRACKING_URI']

        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(exp_name)

        self.mlflow_run = mlflow.start_run(run_name=exp_name)

        if tensorboard_hparams is not None:
            for param_name, param_value in tensorboard_hparams.items():
                mlflow.log_param(param_name, param_value)

        self._url = None

    @property
    def url(self):
        if self._url is None:

            tracking_uri = mlflow.get_tracking_uri()
            current_experiment_id = mlflow.active_run().info.experiment_id
            # Construct the experiment URL (this assumes the default MLflow UI URL structure)
            self._url = f"{tracking_uri}/#/experiments/{current_experiment_id}"

        return self._url

    def add_hparams(self, hparam_dict, metric_dict, **kwargs):
        log_params(hparam_dict)
        for key, value in metric_dict.items():
            log_metric(key, value)

    def add_scalar(self, tag, scalar_value, global_step=None):
        log_metric(tag, scalar_value, step=global_step)

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None):
        for tag, scalar_value in tag_scalar_dict.items():
            full_tag = f"{main_tag}/{tag}"
            log_metric(full_tag, scalar_value, step=global_step)

    def close(self):
        mlflow.end_run()


class MFBeamAlgWrapper(mlflow.pyfunc.PythonModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alg = None

    def load_context(self, context):

        path_to_hparams = context.artifacts['hparams']
        state = context.artifacts['state']

        hparams = beam_path(path_to_hparams).read()

        self.alg = NeuralAlgorithm(hparams)
        self.alg.load_checkpoint(state)

    def predict(self, context, model_input):
        return self.alg.predict(model_input)

    #TODO: follow https://medium.com/@pennyqxr/how-save-and-load-fasttext-model-in-mlflow-format-37e4d6017bf0
    @staticmethod
    def save_model(alg, name, stage=None):

        checkpoint_file = alg.experiment.checkpoints_dir.joinpath(f'checkpoint_mlflow_{alg.epoch + 1:06d}')
        alg.save_checkpoint(checkpoint_file)

        artifacts = {'hparams': str(alg.experiment.experiment_dir.joinpath('hparams.yaml')),
                     'state': str(checkpoint_file)}
        with mlflow.start_run() as run:
            mlflow.pyfunc.log_model(
                artifact_path=str(checkpoint_file.parent.joinpath(checkpoint_file.stem)),
                python_model=alg,
                code_path=[str(alg.experiment.source_dir)],
                artifacts=artifacts,
                registered_model_name=f"{name}/{stage}",

            )

    @staticmethod
    def load_model(name, stage=None):

        if stage is None:
            stage = mlflow.tracking.MlflowClient().get_latest_versions(name, stages=['Production'])[0].version

        loaded_model = mlflow.pyfunc.load_model(model_uri=f"models:/{name}/{stage}")
        return loaded_model