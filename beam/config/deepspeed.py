import os
from collections import defaultdict
from .core_config import BeamConfig, BeamParam
from ..base import base_paths
from ..path import beam_path


def deepspeed_dtype_mapper(dtype):
    model_mapping = {'float32': 'fp32',
                     'float16': 'fp16',
                     'bfloat16': 'bf16'}
    return model_mapping[dtype]


class DeepspeedConfig(BeamConfig):

    parameters = [
        BeamParam('deepspeed_config', str, None, 'Deepspeed configuration JSON file.'),

        # Optimizer Parameters

        BeamParam('deepspeed_optimizer', str, 'AdamW',
                  'Optimizer type (currently used for deepspeed configuration only) '
                  'Supported optimizers: [Adam, AdamW, Lamb, OneBitAdam, OneBitLamb]'),

        # Scheduler Parameters

        # Automatic mixed precision (AMP) training options

        # ZeRO Optimizations for FP16 Training

        BeamParam('zero_stage', int, 2, 'The ZeRO training stage to use.'),
        BeamParam('stage3_gather_16bit_weights_on_model_save', bool, False,
                  'Whether to gather 16-bit weights on model save in ZeRO stage 3'),

        # Parameter offloading

        BeamParam('offload_param_device', str, None, 'Whether to offload parameters from GPU in ZeRO stage 3: '
                                                     '[cpu, nvme, none]'),
        BeamParam('offload_param_pin_memory', bool, True, 'Whether to pin memory for offloaded parameters'),

        BeamParam('offload_param_nvme_path', str, base_paths.deepspeed_data,
                  'Path to NVMe device for offloaded parameters'),

        # Optimizer offloading

        BeamParam('offload_optimizer_device', str, None,
                  'Whether to offload optimizer states from GPU in ZeRO stages 1/2/3: '
                  '[cpu, nvme, none]'),
        BeamParam('offload_optimizer_pin_memory', bool, True, 'Whether to pin memory for offloaded optimizer states'),

        BeamParam('autotuning', bool, False, 'Whether to use deepspeed autotuning feature.'),

        # Activation Checkpointing

        BeamParam('partition_activations', bool, False,
                  'Enables partition activation when used with model parallelism'),
        BeamParam('cpu_checkpointing', bool, False,
                  'Offloads partitioned activations to CPU if partition_activations is enabled'),
        BeamParam('contiguous_memory_optimization', bool, False,
                  'Copies partitioned activations so that they are contiguous in memory'),
        BeamParam('number_checkpoints', int, None,
                  'Total number of activation checkpoints used to allocate memory buffer '
                  'for contiguous_memory_optimization'),
        BeamParam('synchronize_checkpoint_boundary', bool, False,
                  'Inserts get_accelerator().synchronize() at each checkpoint boundary'),
        BeamParam('profile', bool, False, 'Logs the forward and backward time for each checkpoint function'),

        # Sparse Attention

        # Data Efficiency

        # Data Type options
        BeamParam('grad_accum_dtype', str, None, 'The data type for gradient accumulation.'
                                                 'Supported types: [float32, float16, bfloat16]'),
    ]


def recursive_dict_update(d, u):
    '''

    Merge two dicts recursively and update the values of the first one with the values of the second one.
    @param d:
    @param u:
    @return:
    '''
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def recursive_to_dict(obj):
    if isinstance(obj, defaultdict) or isinstance(obj, dict):
        return dict([(k, recursive_to_dict(v)) for k, v in obj.items()])
    return obj


def deepspeed_config_generator(hparams):

    config = defaultdict(dict)
    target = hparams.get('training_framework', 'deepspeed')

    config["train_micro_batch_size_per_gpu"] = hparams.get('batch_size_train') or hparams.get('batch_size')
    config["gradient_accumulation_steps"] = hparams.get('accumulate')

    if hparams.get('zero_stage', 2) is not None:
        config['zero_optimization']['stage'] = hparams.get('zero_stage', 2)
        config['stage3_gather_16bit_weights_on_model_save'] = (
            hparams.get('stage3_gather_16bit_weights_on_model_save', False))

        if hparams.get('offload_param_device', None) is not None:
            config['zero_optimization']['offload_param'] = {}
            config['zero_optimization']['offload_param']['device'] = hparams.get('offload_param_device')
            config['zero_optimization']['offload_param']['pin_memory'] = hparams.get('offload_param_pin_memory', False)

        if hparams.get('offload_optimizer_device', None) is not None:
            config['zero_optimization']['offload_optimizer'] = {}
            config['zero_optimization']['offload_optimizer']['device'] = hparams.get('offload_optimizer_device')
            config['zero_optimization']['offload_optimizer']['pin_memory'] = hparams.get('offload_optimizer_pin_memory', False)

    # optimizer
    if target == 'deepspeed':
        config['optimizer']['type'] = hparams.get('deepspeed_optimizer', 'AdamW')
        config['optimizer']['params'] = {'lr': hparams.get('lr-dense')}
        if hparams.get('weight_decay', None) is not None:
            config['optimizer']['params']['weight_decay'] = hparams.get('weight_decay')
        if 'adam' in config['optimizer']['type'].lower():
            config['optimizer']['params']['betas'] = [hparams.get('momentum', 0.8), hparams.get('beta2', 0.999)]
            config['optimizer']['params']['eps'] = hparams.get('eps', 1e-8)

    # activation_checkpointing
    config['activation_checkpointing']['partition_activations'] = (
        hparams.get('partition_activations', False))
    config['activation_checkpointing']['cpu_checkpointing'] = hparams.get('cpu_checkpointing', False)
    config['activation_checkpointing']['contiguous_memory_optimization'] = (
        hparams.get('contiguous_memory_optimization', False))
    config['activation_checkpointing']['number_checkpoints'] = hparams.get('number_checkpoints', None)
    config['activation_checkpointing']['synchronize_checkpoint_boundary'] = (
        hparams.get('synchronize_checkpoint_boundary', False))
    config['activation_checkpointing']['profile'] = hparams.get('profile', False)

    # autotuning
    if hparams.get('autotuning', False):
        config['autotuning']['enabled'] = True

    model_dtype = deepspeed_dtype_mapper(hparams.get('model_dtype', 'float32'))

    config['fp16']['enabled'] = model_dtype == 'fp16'
    config['bf16']['enabled'] = model_dtype == 'bf16'

    # steps_per_print
    if target != 'accelerate':
        epoch_length_train = hparams.get('epoch_length_train', None)
        epoch_length_eval = hparams.get('epoch_length_eval', None)
        if epoch_length_train is not None and epoch_length_eval is not None:
            config['steps_per_print'] = epoch_length_train + epoch_length_eval

    # update config with deepspeed_config (deepspeed_config overrides all other parameters)
    hparams_dict = recursive_to_dict(config)

    if hparams.get('deepspeed_config', None) is not None:
        config_file_dict = beam_path(hparams.get('deepspeed_config')).read()
        recursive_dict_update(hparams_dict, config_file_dict)

    return hparams_dict
