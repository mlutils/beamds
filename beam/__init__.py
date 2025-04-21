import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

__all__ = ['UniversalBatchSampler', 'UniversalDataset',
           'Experiment', 'nn_algorithm_generator',
           'NeuralAlgorithm',
           'LinearNet', 'PackedTensor', 'copy_network', 'reset_network', 'DataTensor', 'BeamOptimizer', 'BeamScheduler',
           'BeamNN',
           'BeamData',
           'slice_to_index', 'beam_device', 'as_tensor', 'batch_augmentation', 'as_numpy', 'DataBatch', 'beam_hash',
           'UniversalConfig', 'beam_arguments', 'BeamConfig', 'BeamParam',
           'check_type', 'Timer',
           'beam_logger', 'beam_kpi', 'logger',
           'beam_path', 'beam_key', 'pretty_format_number', 'resource',
           'tqdm', 'Transformer', 'Processor',
           'parallel', 'task', 'this_dir', 'cwd', 'chdir',
           # Orchestration
           'BeamDeploy', 'BeamK8S', 'BeamPod', 'K8SUnits', 'K8SConfig', 'RayClusterConfig',
           'ServeClusterConfig', 'ServeCluster', 'RayCluster', 'deploy_server',
           # do not autoreaload these modules
           # 'beam_server', 'beam_client',
           ]


from ._version import __version__
from .logging import beam_logger

from .config import BeamConfig
conf = BeamConfig(silent=True, load_config_files=False, load_script_arguments=True)

log_file_generated = False

# Initialize timer with beam_logger
def initialize_timer():
    from functools import partial
    from .utils import Timer
    from .logging import beam_logger
    return partial(Timer, logger=beam_logger)


def __getattr__(name):

    global log_file_generated
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
    elif name == 'nn_algorithm_generator':
        from .experiment import nn_algorithm_generator
        return nn_algorithm_generator
    elif name == 'NeuralAlgorithm':
        from .algorithm import NeuralAlgorithm
        return NeuralAlgorithm
    elif name == 'LinearNet':
        from .nn import LinearNet
        return LinearNet
    elif name == 'PackedTensor':
        from .nn import PackedTensor
        return PackedTensor
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
        if not log_file_generated:
            from .path import beam_path
            path = beam_path(conf.beam_logs_path)
            path.mkdir()
            t = beam_logger.timestamp()
            program = sys.argv[0].split('/')[-1].split('.')[0]
            path = path.joinpath(f"{program}-{t}.log")
            beam_logger.add_default_file_handler(path)
            beam_logger.info(f"Beam logger ({__version__}): logs are saved to {path}")
            beam_logger.debug("to stop logging to this file use beam_logger.remove_default_handlers()")
            log_file_generated = True
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
    elif name == 'this_dir':
        from .resources import this_dir
        return this_dir
    elif name == 'cwd':
        from .resources import cwd
        return cwd
    elif name == 'chdir':
        from .resources import chdir
        return chdir
    # Orchestration
    elif name == 'BeamDeploy':
        from .orchestration import BeamDeploy
        return BeamDeploy
    elif name == 'BeamK8S':
        from .orchestration import BeamK8S
        return BeamK8S
    elif name == 'BeamPod':
        from .orchestration import BeamPod
        return BeamPod
    elif name == 'K8SUnits':
        from .orchestration import K8SUnits
        return K8SUnits
    elif name == 'K8SConfig':
        from .orchestration import K8SConfig
        return K8SConfig
    elif name == 'RayClusterConfig':
        from .orchestration import RayClusterConfig
        return RayClusterConfig
    elif name == 'ServeClusterConfig':
        from .orchestration import ServeClusterConfig
        return ServeClusterConfig
    elif name == 'ServeCluster':
        from .orchestration import ServeCluster
        return ServeCluster
    elif name == 'RayCluster':
        from .orchestration import RayCluster
        return RayCluster
    elif name == 'deploy_server':
        from .orchestration import deploy_server
        return deploy_server
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")


# Explicit imports for IDE
if len([]):
    from .utils import tqdm_beam as tqdm
    from .dataset import UniversalBatchSampler, UniversalDataset
    from .experiment import Experiment, nn_algorithm_generator
    from .algorithm import NeuralAlgorithm
    from .nn import LinearNet, PackedTensor, copy_network, reset_network, DataTensor, BeamOptimizer, BeamScheduler, BeamNN
    from .data import BeamData
    from .utils import slice_to_index, beam_device, as_tensor, batch_augmentation, as_numpy, DataBatch, beam_hash
    from .config import UniversalConfig, beam_arguments, BeamConfig, BeamParam
    from .utils import check_type, Timer, pretty_format_number
    from .logging import beam_logger, beam_kpi, beam_logger as logger
    from .path import beam_path, beam_key
    from .serve import beam_server, beam_client
    from ._version import __version__
    from .resources import resource, this_dir, cwd, chdir
    from .transformer import Transformer
    from .processor import Processor
    from .concurrent import parallel, task
    from .orchestration import (BeamDeploy, BeamK8S, BeamPod, K8SUnits, K8SConfig, RayClusterConfig,
                                ServeClusterConfig, ServeCluster, RayCluster, deploy_server)
