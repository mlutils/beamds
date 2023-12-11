import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from .utils import tqdm_beam as tqdm

try:
    import torch
    has_torch = True
except ImportError:
    has_torch = False

if has_torch:
    from .dataset import UniversalBatchSampler, UniversalDataset
    from .experiment import Experiment, beam_algorithm_generator
    from .core import Algorithm
    from .nn import LinearNet, PackedSet, copy_network, reset_network
    from .tensor import DataTensor
    from .nn import BeamOptimizer, BeamScheduler
    from .data import BeamData
    from .packed_folds import PackedFolds
    from .utils import slice_to_index, beam_device, as_tensor, batch_augmentation, as_numpy, DataBatch, beam_hash

from .config import basic_beam_parser, beam_arguments, BeamHparams
from .utils import check_type, Timer
from .logger import beam_logger, beam_kpi

from functools import partial
Timer = partial(Timer, logger=beam_logger)

from .path import beam_path, beam_key
from .serve import beam_server, beam_client
from ._version import __version__
from .resource import resource

