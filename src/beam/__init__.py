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
    from .algorithm import Algorithm
    from .model import LinearNet, PackedSet, copy_network, reset_network
    from .tensor import DataTensor
    from .optim import BeamOptimizer, BeamScheduler
    from .data import BeamData
    from .packed_folds import PackedFolds
    from .utils import slice_to_index, beam_device, as_tensor, batch_augmentation, as_numpy, DataBatch


from .config import get_beam_parser, beam_arguments, BeamHparams
from .config import boolean_feature as beam_boolean_feature

from .utils import check_type

from .logger import beam_logger, beam_kpi, Timer
from .path import beam_path, beam_key
from ._version import __version__

from .resource import resource

# from .llm import beam_llm
# from .ssl import BeamSimilarity, Similarities, BeamSSL, BYOL, BeamVICReg, BarlowTwins, VICReg, SimCLR, SimSiam
# from .server import BeamServer, BeamClient
