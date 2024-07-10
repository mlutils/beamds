import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

__all__ = ['UniversalBatchSampler', 'UniversalDataset',
           'Experiment', 'beam_algorithm_generator',
           'NeuralAlgorithm',
           'LinearNet', 'PackedSet', 'copy_network', 'reset_network', 'DataTensor', 'BeamOptimizer', 'BeamScheduler',
           'BeamNN',
           'BeamData',
           'slice_to_index', 'beam_device', 'as_tensor', 'batch_augmentation', 'as_numpy', 'DataBatch', 'beam_hash',
           'UniversalConfig', 'beam_arguments', 'BeamConfig', 'BeamParam',
           'check_type', 'Timer',
           'beam_logger', 'beam_kpi', 'logger',
           'beam_path', 'beam_key', 'pretty_format_number', 'resource',
           'tqdm', 'Transformer', 'Processor',
           'parallel', 'task'
           ]


from ._version import __version__

from .config import BeamConfig
from .logging import beam_logger

conf = BeamConfig(silent=True, load_config_files=False, load_script_arguments=False)
if conf.debug:
    beam_logger.debug_mode()


# Initialize timer with beam_logger
def initialize_timer():
    from functools import partial
    from .utils import Timer
    from .logging import beam_logger
    return partial(Timer, logger=beam_logger)


def __getattr__(name):
    if name in ['tqdm', 'tqdm_beam']:
        from .utils import tqdm_beam
        return tqdm_beam
    elif name == 'UniversalBatchSampler':
        from .dataset import UniversalBatchSampler
        return UniversalBatchSampler
    elif name == 'UniversalDataset':
        from .dataset import UniversalDataset
        return UniversalDataset
    elif name == 'Experiment':
        from .experiment import Experiment
        return Experiment
    elif name == 'beam_algorithm_generator':
        from .experiment import beam_algorithm_generator
        return beam_algorithm_generator
    elif name == 'NeuralAlgorithm':
        from .algorithm import NeuralAlgorithm
        return NeuralAlgorithm
    elif name == 'LinearNet':
        from .nn import LinearNet
        return LinearNet
    elif name == 'PackedSet':
        from .nn import PackedSet
        return PackedSet
    elif name == 'copy_network':
        from .nn import copy_network
        return copy_network
    elif name == 'reset_network':
        from .nn import reset_network
        return reset_network
    elif name == 'DataTensor':
        from .nn import DataTensor
        return DataTensor
    elif name == 'BeamOptimizer':
        from .nn import BeamOptimizer
        return BeamOptimizer
    elif name == 'BeamScheduler':
        from .nn import BeamScheduler
        return BeamScheduler
    elif name == 'BeamNN':
        from .nn import BeamNN
        return BeamNN
    elif name == 'BeamData':
        from .data import BeamData
        return BeamData
    elif name == 'beam_key':
        from .path import beam_key
        return beam_key
    elif name == 'slice_to_index':
        from .utils import slice_to_index
        return slice_to_index
    elif name == 'beam_device':
        from .utils import beam_device
        return beam_device
    elif name == 'as_tensor':
        from .utils import as_tensor
        return as_tensor
    elif name == 'batch_augmentation':
        from .utils import batch_augmentation
        return batch_augmentation
    elif name == 'as_numpy':
        from .utils import as_numpy
        return as_numpy
    elif name == 'DataBatch':
        from .utils import DataBatch
        return DataBatch
    elif name == 'beam_hash':
        from .utils import beam_hash
        return beam_hash
    elif name == 'UniversalConfig':
        from .config import UniversalConfig
        return UniversalConfig
    elif name == 'beam_arguments':
        from .config import beam_arguments
        return beam_arguments
    elif name == 'BeamConfig':
        from .config import BeamConfig
        return BeamConfig
    elif name == 'BeamParam':
        from .config import BeamParam
        return BeamParam
    elif name == 'check_type':
        from .utils import check_type
        return check_type
    elif name == 'Timer':
        return initialize_timer()
    elif name in ['beam_logger', 'logger']:
        from .logging import beam_logger
        return beam_logger
    elif name == 'beam_kpi':
        from .logging import beam_kpi
        return beam_kpi
    elif name == 'beam_path':
        from .path import beam_path
        return beam_path
    elif name == 'pretty_format_number':
        from .utils import pretty_format_number
        return pretty_format_number
    elif name == 'beam_server':
        from .serve import beam_server
        return beam_server
    elif name == 'beam_client':
        from .serve import beam_client
        return beam_client
    elif name == 'resource':
        from .resources import resource as bea_resource
        return bea_resource
    elif name == 'Transformer':
        from .transformer import Transformer
        return Transformer
    elif name == 'Processor':
        from .processor import Processor
        return Processor
    elif name == 'parallel':
        from .concurrent import parallel
        return parallel
    elif name == 'task':
        from .concurrent import task
        return task
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")


# Explicit imports for IDE
if len([]):
    from .utils import tqdm_beam as tqdm
    from .dataset import UniversalBatchSampler, UniversalDataset
    from .experiment import Experiment, beam_algorithm_generator
    from .algorithm import NeuralAlgorithm
    from .nn import LinearNet, PackedSet, copy_network, reset_network, DataTensor, BeamOptimizer, BeamScheduler, BeamNN
    from .data import BeamData
    from .utils import slice_to_index, beam_device, as_tensor, batch_augmentation, as_numpy, DataBatch, beam_hash
    from .config import UniversalConfig, beam_arguments, BeamConfig, BeamParam
    from .utils import check_type, Timer, pretty_format_number
    from .logging import beam_logger, beam_kpi, beam_logger as logger
    from .path import beam_path, beam_key
    from .serve import beam_server, beam_client
    from ._version import __version__
    from .resources import resource
    from .transformer import Transformer
    from .processor import Processor
    from .concurrent import parallel, task
