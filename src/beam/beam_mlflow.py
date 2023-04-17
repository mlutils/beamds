import mlflow.pyfunc
from algorithm import Algorithm
from path import beam_path
import mlflow


class MFBeamAlgWrapper(mlflow.pyfunc.PythonModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alg = None

    def load_context(self, context):

        path_to_hparams = context.artifacts['hparams']
        path_to_state = context.artifacts['state']

        hparams = beam_path(path_to_hparams).read()

        self.alg = Algorithm(hparams)
        self.alg.load_checkpoint(path_to_state)

    def predict(self, context, model_input):
        return self.alg.predict(model_input)

    #TODO: follow https://medium.com/@pennyqxr/how-save-and-load-fasttext-model-in-mlflow-format-37e4d6017bf0
    @staticmethod
    def save_model(alg, name, stage=None):

        checkpoint_file = alg.experiment.checkpoints_dir.joinpath(f'checkpoint_mlflow_{alg.epoch + 1:06d}')
        alg.save_checkpoint(checkpoint_file)

        artifacts = {'hparams': str(alg.experiment.root.joinpath('hparams.pkl')),
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