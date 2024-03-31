import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

from .utils import tqdm_beam as tqdm

__all__ = ['UniversalBatchSampler', 'UniversalDataset',
           'Experiment', 'beam_algorithm_generator',
           'NeuralAlgorithm',
           'LinearNet', 'PackedSet', 'copy_network', 'reset_network', 'DataTensor', 'BeamOptimizer', 'BeamScheduler',
           'BeamNN',
           'BeamData',
           'slice_to_index', 'beam_device', 'as_tensor', 'batch_augmentation', 'as_numpy', 'DataBatch', 'beam_hash',
           'UniversalConfig', 'beam_arguments', 'BeamConfig', 'BeamParam',
           'check_type', 'Timer',
           'beam_logger', 'beam_kpi',
           'beam_path', 'beam_key',
           'beam_server', 'beam_client',
           '__version__',
           'resource',
           'KeysConfig',
           'tqdm',
           ]


try:
    import torch
    has_torch = True
except ImportError:
    has_torch = False

if has_torch:
    from .dataset import UniversalBatchSampler, UniversalDataset
    from .experiment import Experiment, beam_algorithm_generator
    from .algorithm import NeuralAlgorithm
    from .nn import LinearNet, PackedSet, copy_network, reset_network, DataTensor, BeamOptimizer, BeamScheduler, BeamNN
    from .data import BeamData
    from .utils import slice_to_index, beam_device, as_tensor, batch_augmentation, as_numpy, DataBatch, beam_hash

from .config import UniversalConfig, beam_arguments, BeamConfig, BeamParam
from .utils import check_type, Timer
from .logger import beam_logger, beam_kpi

from functools import partial
Timer = partial(Timer, logger=beam_logger)

from .path import beam_path, beam_key
from .serve import beam_server, beam_client
from ._version import __version__
from .resource import resource

from .config import KeysConfig
beam_key.set_hparams(KeysConfig(silent=True, strict=True))

