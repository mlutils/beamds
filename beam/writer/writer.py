import os
from typing import List

from ..base import BeamBase
from ..path import beam_path
from ..logging import beam_logger as logger
from ..utils import cached_property


class BeamWriter(BeamBase):

    def __init__(self, *args, code_dir=None, **kwargs):
        super(BeamWriter, self).__init__(*args, **kwargs)
        self.code_dir = code_dir

    @cached_property
    def client(self):
        raise NotImplementedError("This method must be implemented by the subclass")

    @classmethod
    def from_url(cls, url):
        raise NotImplementedError("This method must be implemented by the subclass")

    @cached_property
    def root(self):
        raise NotImplementedError("This method must be implemented by the subclass")

    def set_name(self, name):
        raise NotImplementedError("This method must be implemented by the subclass")

    def add_tag(self, tag):
        raise NotImplementedError("This method must be implemented by the subclass")

    def add_hparams(self, hparam_dict, metric_dict, **kwargs):
        self.client.add_hparams(hparam_dict, metric_dict, **kwargs)

    def add_scalar(self, tag, scalar_value, global_step=None):
        self.client.add_scalar(tag, scalar_value, global_step)

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None):
        self.client.add_scalars(main_tag, tag_scalar_dict, global_step)

    def add_histogram(self, tag, values, global_step=None, bins='tensorflow', walltime=None, max_bins=None):
        self.client.add_histogram(tag, values, global_step, bins, walltime, max_bins)

    def add_image(self, tag, img_tensor, global_step=None, walltime=None, dataformats='CHW'):
        self.client.add_image(tag, img_tensor, global_step, walltime, dataformats)

    def add_images(self, tag, img_tensor, global_step=None, walltime=None, dataformats='NCHW'):
        self.client.add_images(tag, img_tensor, global_step, walltime, dataformats)

    def add_figure(self, tag, figure, global_step=None, close=True, walltime=None):
        self.client.add_figure(tag, figure, global_step, close, walltime)

    def add_video(self, tag, vid_tensor, global_step=None, fps=4, walltime=None):
        self.client.add_video(tag, vid_tensor, global_step, fps, walltime)

    def add_audio(self, tag, snd_tensor, global_step=None, sample_rate=44100, walltime=None):
        self.client.add_audio(tag, snd_tensor, global_step, sample_rate, walltime)

    def add_text(self, tag, text_string, global_step=None, walltime=None):
        self.client.add_text(tag, text_string, global_step, walltime)

    def add_graph(self, model, input_to_model=None, verbose=False):
        self.client.add_graph(model, input_to_model, verbose)

    def add_embedding(self, mat, metadata=None, label_img=None, global_step=None, tag='default', metadata_header=None):
        self.client.add_embedding(mat, metadata, label_img, global_step, tag, metadata_header)

    def add_pr_curve(self, tag, labels, predictions, global_step=None, num_thresholds=127, weights=None, walltime=None):
        self.client.add_pr_curve(tag, labels, predictions, global_step, num_thresholds, weights, walltime)

    def add_custom_scalars(self, layout):
        self.client.add_custom_scalars(layout)

    def add_mesh(self, tag, vertices, colors=None, faces=None, config_dict=None, global_step=None):
        self.client.add_mesh(tag, vertices, colors, faces, config_dict, global_step)

    def add_pr_curve_raw(self, tag, true_positive_counts, false_positive_counts, true_negative_counts,
                         false_negative_counts, precision, recall, global_step=None, num_thresholds=127, weights=None,
                         walltime=None):
        self.client.add_pr_curve_raw(tag, true_positive_counts, false_positive_counts, true_negative_counts,
                                     false_negative_counts, precision, recall, global_step, num_thresholds, weights,
                                     walltime)

    def add_onnx_graph(self, model):
        self.client.add_onnx_graph(model)

    def add_openvino_graph(self, model):
        self.client.add_openvino_graph(model)

    def add_custom_scalars_marginchart(self, tags: List[str], category: str = 'default', title: str = 'untitled'):
        self.client.add_custom_scalars_marginchart(tags, category, title)

    def add_custom_scalars_multilinechart(self, tags: List[str], category: str = 'default', title: str = 'untitled'):
        self.client.add_custom_scalars_multilinechart(tags, category, title)

    def add_histogram_raw(self, tag, min, max, num, sum, sum_squares, bucket_limits, bucket_counts, global_step=None,
                          walltime=None):
        self.client.add_histogram_raw(tag, min, max, num, sum, sum_squares, bucket_limits, bucket_counts, global_step,
                                     walltime)


class TensorboardWriter(BeamWriter):

    def __init__(self, *args, path=None, **kwargs):
        super(TensorboardWriter, self).__init__(*args, **kwargs)
        self.path = path

    @cached_property
    def client(self):
        from tensorboardX import SummaryWriter
        return SummaryWriter(log_dir=self.path)

    @classmethod
    def from_url(cls, url):
        path = beam_path(url).str
        return cls(path=path)

    @cached_property
    def root(self):
        return beam_path(self.path)

    def close(self):
        self.client.close()

    def set_name(self, name):
        logger.warning("Setting name to experiment is not supported by Tensorboard")

    def add_tag(self, tag):
        logger.warning("Adding tags to experiment is not supported by Tensorboard")


class CometWriter(BeamWriter):

    def __init__(self, *args, hostname=None, tls=None, api_key=None, experiment_name=None,
                 project_name=None, **kwargs):
        super(CometWriter, self).__init__(*args, **kwargs)
        if experiment_name is not None:
            path = f'{hostname}/{project_name}/{experiment_name}'
        else:
            path = f'{hostname}/{project_name}'
        self.resource = beam_path(f"comet://{path}?tls={tls}&api_key={api_key}")

    @cached_property
    def client(self):

        from tensorboardX import SummaryWriter

        if not self.resource.exists():

            if self.code_dir is not None:
                os.environ['COMET_GIT_DIRECTORY'] = self.code_dir
                log_code = True
            else:
                log_code = False

            logger.info("Logging this experiment to comet.ml")

            client = SummaryWriter(comet_config={'api_key': self.resource.access_key,
                                                 'project_name': self.hparams.project_name,
                                                 'log_code': log_code,
                                                 'workspace': self.hparams.comet_workspace,
                                                 'disabled': False})
            comet_logger = client._get_comet_logger()._experiment
            experiment_name = comet_logger.name
            self.resource.joinpath(experiment_name)

        else:
            from comet_ml import ExistingExperiment
            client = SummaryWriter(comet_config={'disabled': False})
            comet_logger = client._get_comet_logger()
            comet_logger._logging = True
            comet_logger._experiment = ExistingExperiment(api_key=self.resource.access_key,
                                                          experiment_key=self.resource.experiment_key)

        return client

    @cached_property
    def _experiment(self):
        return self.client._get_comet_logger()._experiment

    @classmethod
    def from_url(cls, url):
        path = beam_path(url)
        assert path.scheme == 'comet', "Invalid URL"
        assert path.level in [2, 3], ("Requires workspace and project and experiment name in the URL "
                                      "(without artifact path)")
        return cls(api_key=path.access_key, hostname=path.hostname, project_name=path.project_name,
                   experiment_name=path.experiment_name, tls=path.tls)

    @cached_property
    def root(self):
        _ = self.client
        return self.resource

    def set_name(self, name):
        self._experiment.set_name(name)

    def add_tag(self, tag):
        self._experiment.add_tag(tag)

    def add_graph(self, model, input_to_model=None, verbose=False):
        self._experiment.set_model_graph(str(model))


