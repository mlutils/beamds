import argparse
import os
import math
from .core_config import BeamConfig, BeamParam
from .deepspeed import DeepspeedConfig
from .utils import _beam_arguments, empty_beam_parser
from ..base import base_paths


class DeviceConfig(BeamConfig):

    parameters = [
        BeamParam('device', str, '0', 'GPU Number or cpu/cuda string'),
        BeamParam('device_list', list, None,
                  'Set GPU priority for parallel execution e.g. --device-list 2 1 3 will use GPUs 2 and 1 '
                  'when passing --n-gpus=2 and will use GPUs 2 1 3 when passing --n-gpus=3. '
                  'If None, will use an ascending order starting from the GPU passed in the --device parameter. '
                  'e.g. when --device=1 will use GPUs 1,2,3,4 when --n-gpus=4'),
        BeamParam('n_gpus', int, 1, 'Number of parallel gpu workers. Set <=1 for single process'),
    ]


class CacheConfig(BeamConfig):

    parameters = [
        BeamParam('cache_depth', int, None, 'The depth of the cache'),
        BeamParam('cache_path', str, None, 'The path to the cache (if None, the cache is stored in memory)'),
        BeamParam('cache_exception_keys', list, None, 'The keys to exclude from the cache'),
        BeamParam('cache_store_suffix', str, None, 'The suffix to add to the stored file '
                                                   '(if None, the cache is stored as BeamData)'),
        BeamParam('silent_cache', bool, False, 'Whether to log cache operations'),
    ]


class NNModelConfig(BeamConfig):
    parameters = [
        BeamParam('init', str, 'ortho', 'Initialization method [ortho|N02|xavier|]', tags='tune'),
    ]


class SchedulerConfig(BeamConfig):
    
    parameters = [
        BeamParam('scheduler_steps', str, 'epoch', 'When to apply schedulers steps [epoch|iteration|none]: '
                                                   'each epoch or each iteration. '
                                                   'Use none to avoid scheduler steps or to use your own custom steps policy'),
    
        BeamParam('scheduler', str, None, 'Build BeamScheduler. Supported schedulers: '
                                          '[one_cycle,reduce_on_plateau,cosine_annealing]', tags='tune'),
    
        BeamParam('cycle_base_momentum', float, .85, 'The base momentum in one-cycle optimizer', tags='tune'),
        BeamParam('cawr_t0', int, 10, ' Number of iterations for the first restart in CosineAnnealingWarmRestarts scheduler',
                    tags='tune'),
        BeamParam('cawr_tmult', int, 1, ' A factor increases Ti after a restart in CosineAnnealingWarmRestarts scheduler',
                    tags='tune'),
        BeamParam('scheduler_factor', float, math.sqrt(.1), 'The factor to reduce lr in schedulers such as ReduceOnPlateau',
                    tags='tune'),
        BeamParam('scheduler_patience', int, 10, 'Patience for the ReduceOnPlateau scheduler', tags='tune'),
        BeamParam('scheduler_warmup', float, 5, 'Scheduler\'s warmup factor (in epochs)', tags='tune'),
        BeamParam('cycle_max_momentum', float, .95, 'The maximum momentum in one-cycle scheduler', tags='tune'),
    ]


class DatasetConfig(BeamConfig):

    parameters = [
        BeamParam('split_dataset_seed', int, 5782, 'Seed dataset split (set to zero to get random split)'),
        BeamParam('test_size', float, .2, 'Test set percentage'),
        BeamParam('validation_size', float, .2, 'Validation set percentage'),
        BeamParam('stratify_dataset', bool, False, 'Stratify the dataset split by the labels'),
        BeamParam('dataset_time_index', str, None, 'The time index to use for time-based splits'),
        BeamParam('test_split_method', str, 'uniform', 'The method to split the test set [uniform|time_based]'),
    ]


class SamplerConfig(BeamConfig):

    parameters = [
        BeamParam('oversampling_factor', float, .0, 'A factor [0, 1] that controls how much to oversample where '
                                                    '0-no oversampling and 1-full oversampling. Set 0 for no oversampling',
                  tags='tune'),
        BeamParam('expansion_size', int, int(1e7), 'largest expanded index size for oversampling'),
        BeamParam('dynamic_sampler', bool, False, 'Whether to use a dynamic sampler (mainly for rl/optimization)'),
        BeamParam('buffer_size', int, None, 'Maximal Dataset size in dynamic problems', tags='tune'),
        BeamParam('probs_normalization', str, 'sum', 'Sampler\'s probabilities normalization method [sum/softmax]'),
        BeamParam('sample_size', int, 100000, 'Periodic sample size for the dynamic sampler'),
    ]


class OptimizerConfig(BeamConfig):

    parameters = [
        BeamParam('weight_decay', float, 0., 'L2 regularization coefficient for dense optimizers', tags='tune'),
        BeamParam('eps', float, 1e-4, 'Adam\'s epsilon parameter', tags='tune'),
        BeamParam('momentum', float, .9, 'The momentum and Adam\'s β1 parameter', tags='tune'),
        BeamParam('beta2', float, .999, 'Adam\'s β2 parameter', tags='tune'),
        BeamParam('clip_gradient', float, 0., 'Clip Gradient L2 norm', tags='tune'),
        BeamParam('accumulate', int, 1, 'Accumulate gradients for this number of backward iterations', tags='tune'),
    ]


class SWAConfig(BeamConfig):

    parameters = [
        BeamParam('swa', float, None, 'SWA period. If float it is a fraction of the total number of epochs. '
                                      'If integer, it is the number of SWA epochs.'),
        BeamParam('swa_lr', float, 0.05, 'The SWA learning rate', tags='tune'),
        BeamParam('swa_anneal_epochs', int, 10, 'The SWA lr annealing period', tags='tune'),
    ]


class FederatedTrainingConfig(DeviceConfig):

    parameters = [
        BeamParam('mp_ip', str, 'localhost', 'IP to be used for multiprocessing'),
        BeamParam('mp_port', str, None, 'Port to be used for multiprocessing'),
        BeamParam('n_cpus_per_worker', int, 6, 'Number of cpus to use in each worker'),
        BeamParam('n_gpus_per_worker', int, 1, 'Number of gpus to use in each worker'),
        BeamParam('distributed_backend', str, 'nccl', 'The distributed backend to use. '
                                                    'Supported backends: [nccl, gloo, mpi]'),
        BeamParam('mp_context', str, 'spawn', 'The multiprocessing context to use'),
        BeamParam('kv_store', str, 'tcp', 'The key-value store to use [tcp|file|hash]'),
        BeamParam('kv_store_path', str, None, 'The path to the key-value store file'),
        BeamParam('kv_store_timeout', float, 300., 'The timeout for the key-value store'),
        BeamParam('kv_store_port', str, None, 'The port to use for the key-value store'),
        BeamParam('federated_runner', bool, False, 'Use the new federated runner for distributed training'),

    ]


class AccelerateConfig(DeepspeedConfig, FederatedTrainingConfig):
    # accelerate parameters
    # based on https://huggingface.co/docs/accelerate/v0.24.0/en/package_reference/accelerator#accelerate.Accelerator

    # boolean_feature(parser, "deepspeed-dataloader", False,
    #                 "Use optimized deepspeed dataloader instead of native pytorch dataloader")

    parameters = [
        BeamParam('device_placement', bool, False, 'Whether or not the accelerator should put objects on device'),
        BeamParam('split_batches', bool, False, 'Whether or not the accelerator should split the batches '
                                                'yielded by the dataloaders across the devices'),
    ]


class NNCompilerConfig(BeamConfig):

    """
    For torch compile see:
    https://pytorch.org/docs/stable/generated/torch.compile.html

    For torch jit see:
    https://pytorch.org/docs/stable/generated/torch.jit.trace.html
    """

    parameters = [
        BeamParam('compile_fullgraph', bool, False, 'Whether it is ok to break model into several subgraphs'),
        BeamParam('compile_dynamic', bool, None, 'Use dynamic shape tracing. When this is True, '
                                                  'we will up-front attempt to generate a kernel that is as dynamic as '
                                                  'possible to avoid recompilations when sizes change.'),
        BeamParam('compile_backend', str, 'inductor', 'The backend to use for compilation [inductor|torch]'),
        BeamParam('compile_mode', str, 'default', '[default|reduce-overhead|max-autotune|max-autotune-no-cudagraphs],'
                                                  ' see https://pytorch.org/docs/stable/generated/torch.compile.html'),
        BeamParam('compile_options', dict, None, 'Additional options for the compiler'),


        BeamParam('jit_check_trace', bool, True, 'Check if the same inputs run through traced code produce the same '
                                                 'outputs'),
        BeamParam('jit_check_inputs', list, None, 'A list of tuples of input arguments that should be used to check the '
                                                  'trace against what is expected.'),
        BeamParam('jit_check_tolerance', float, 1e-5, 'Floating-point comparison tolerance to use in the checker '
                                                      'procedure.'),
        BeamParam('jit_strict', bool, True, 'Only turn this off when you want the tracer to record your mutable container '
                                            'types'),
    ]


class NNTrainingConfig(NNModelConfig, SchedulerConfig, AccelerateConfig,
                       SWAConfig, OptimizerConfig, DatasetConfig, SamplerConfig, NNCompilerConfig):

    parameters = [

        BeamParam('scale_epoch_by_batch_size', bool, True,
                  'When True: epoch length corresponds to the number of examples sampled from the dataset in each epoch '
                  'When False: epoch length corresponds to the number of forward passes in each epoch'),

        BeamParam('model_dtype', str, 'float32', 'dtype, both for automatic mixed precision and accelerate. '
                                                 'Supported dtypes: [float32, float16, bfloat16]', tags=['tune', 'model']),
        BeamParam('total_steps', int, int(1e6), 'Total number of environment steps', tags='tune'),
        BeamParam('epoch_length', int, None, 'Length of train+eval epochs '
                                             '(if None - it is taken from epoch_length_train/epoch_length_eval arguments)',
                  tags='tune'),
        BeamParam('train_on_tail', bool, False, 'Should the last (incomplete) batch be included in the training epoch '
                                                '(useful for small datasets but not time efficient)'),
        BeamParam('epoch_length_train', int, None, 'Length of each epoch (if None - it is the dataset[train] size)', tags='tune'),
        BeamParam('epoch_length_eval', int, None, 'Length of each evaluation epoch (if None - it is the dataset[validation] size)',
                    tags='tune'),
        BeamParam('n_epochs', int, None, 'Number of epochs, if None, '
                                         'it uses the total steps to determine the number of iterations', tags='tune'),
        BeamParam('batch_size', int, 256, 'Batch Size', tags='tune'),
        BeamParam('batch_size_train', int, None, 'Batch Size for training iterations', tags='tune'),
        BeamParam('batch_size_eval', int, None, 'Batch Size for testing/evaluation iterations', tags='tune'),
        BeamParam('reduction', str, 'sum', 'whether to sum loss elements or average them [sum|mean|mean_batch|sqrt|mean_sqrt]',
                    tags='tune'),
        BeamParam(['lr_dense', 'lr'], float, 1e-3, 'learning rate for dense optimizers', tags='tune'),
        BeamParam('lr_sparse', float, 1e-2, 'learning rate for sparse optimizers', tags='tune'),
        BeamParam('stop_at', float, None, 'Early stopping when objective >= stop_at', tags='tune'),
        BeamParam('early_stopping_patience', int, None, 'Early stopping patience in epochs, '
                                                     'stop when current_epoch - best_epoch >= early_stopping_patience',
                    tags='tune'),
    ]


class KeysConfig(BeamConfig):

    parameters = [
        BeamParam('COMET_API_KEY', str, None, 'The comet.ml api key to use for logging'),
        BeamParam('AWS_ACCESS_KEY_ID', str, None, 'The aws access key to use for S3 connections'),
        BeamParam('AWS_SECRET_ACCESS_KEY', str, None, 'The aws private key to use for S3 connections'),
        BeamParam('SSH_PRIVATE_KEY', str, None, 'The ssh secret key to use for ssh connections'),
        BeamParam('OPENAI_API_KEY', str, None, 'The openai api key to use for openai connections'),
        BeamParam('BEAM_USERNAME', str, None, 'The beam username to use for connections like smb/ftp/ssh etc'),
        BeamParam('BEAM_PASSWORD', str, None, 'The beam password to use for connections like smb/ftp/ssh etc'),
        BeamParam('K8S_API_KEY', str, None, 'The k8s api key to use for k8s connections'),
    ]


class DDPConfig(BeamConfig):

    parameters = [
        BeamParam('find_unused_parameters', bool, False, 'For DDP applications: allows running backward on '
                                                         'a subgraph of the model. introduces extra overheads, '
                                                         'so applications should only set find_unused_parameters '
                                                         'to True when necessary'),

        BeamParam('broadcast_buffers', bool, True, 'For DDP applications: Flag that enables syncing (broadcasting) '
                                                   'buffers of the module at beginning of the forward function.'),

        BeamParam('nvlink', bool, False, 'For DDP applications: whether nvlink is available for faster communication'),
    ]


class BeamProjectConfig(BeamConfig):
    parameters = [
        BeamParam('project_name', str, 'beam', 'The name of the beam project'),
        BeamParam('algorithm', str, 'Algorithm', 'algorithm name'),
        BeamParam('identifier', str, 'debug', 'The name of the model to use'),
        BeamParam('logs_path', str, base_paths.projects_experiments, 'Root directory for Logs and results'),
        BeamParam('data_path', str, base_paths.projects_data, 'Where the dataset is located'),
        BeamParam('config_file', str, None, 'The beam config file to use with secret keys'),
        BeamParam('verbosity', str, 'info', 'The verbosity level [debug|info|warning|error]'),
    ]


class ExperimentConfig(BeamProjectConfig, KeysConfig, CacheConfig):
    '''

        Arguments

            global parameters

            These parameters are responsible for which experiment to load or to generate:
            the name of the experiment is <alg>_<identifier>_exp_<num>_<time>
            The possible configurations:
            reload = False, override = True: always overrides last experiment (default configuration)
            reload = False, override = False: always append experiment to the list (increment experiment num)
            reload = True, resume = -1: resume to the last experiment
            reload = True, resume = <n>: resume to the <n> experiment

    '''

    parameters = [

        BeamParam('llm', str, None, 'URI of a Large Language Model to be used in the experiment.'),
        BeamParam('reload', bool, False, 'Load saved model'),
        BeamParam('resume', int, -1, 'Resume experiment number, set -1 for last experiment: active when reload=True'),
        BeamParam('override', bool, False, 'Override last experiment: active when reload=False'),
        BeamParam('reload_checkpoint', str, 'best', 'Which checkpoint to reload [best|last|<epoch>]'),

        BeamParam('cpu_workers', int, 0, 'How many CPUs will be used for the data loading'),
        BeamParam('data_fetch_timeout', float, 0., 'Timeout for the dataloader fetching. '
                                                   'set to 0 for no timeout.'),

        BeamParam('tensorboard', bool, True, 'Log results to tensorboard'),
        BeamParam('mlflow', bool, False, 'Log results to MLFLOW serve'),
        BeamParam('lognet', bool, True, 'Log  networks parameters'),
        BeamParam('deterministic', bool, False, 'Use deterministic pytorch optimization for reproducability '
                                                'when enabling non-deterministic behavior, it sets '
                                                'torch.backends.cudnn.benchmark = True which '
                                                'accelerates the computation'),

        BeamParam('scalene', bool, False, 'Profile the experiment with the Scalene python profiler'),
        BeamParam('safetensors', bool, False, 'Save tensors in safetensors format instead of native torch'),
        BeamParam('store_initial_weights', bool, False, 'Store the network\'s initial weights'),
        BeamParam('store_init_args', bool, True, 'Store the algorithm init args/kwargs for better reloading'),
        BeamParam('copy_code', bool, True, 'Copy the code directory into the experiment directory'),
        BeamParam('restart_epochs_count', bool, True, 'When reloading an algorithm, restart counting epochs from zero '
                                                      '(with respect to schedulers and swa training)', tags='tune'),
        BeamParam('seed', int, 0, 'Seed for reproducability (zero is saved for random seed)'),
        BeamParam('train_timeout', int, None, 'Timeout for the training in seconds. Set to None for no timeout',
                  tags='tune'),

        # results printing and visualization

        BeamParam('log_experiment', bool, True, 'Log experiment to the log directory'),
        BeamParam('print_results', bool, True, 'Print results after each epoch to screen'),
        BeamParam('visualize_weights', bool, False, 'Visualize network weights on tensorboard'),
        BeamParam('enable_tqdm', bool, True, 'Print tqdm progress bar when training'),
        BeamParam('visualize_results_log_base', int, 10, 'log base for the logarithmic based results visualization'),
        BeamParam('tqdm_threshold', float, 10., 'Minimal expected epoch time to print tqdm bar '
                                                'set 0 to ignore and determine tqdm bar with tqdm-enable flag'),
        BeamParam('tqdm_stats', float, 1., 'Take this period to calculate the expected epoch time'),

        BeamParam('visualize_results', str, 'yes', 'when to visualize results on tensorboard '
                                                   '[yes|no|logscale|best|never|final]'),
        BeamParam('store_results', str, 'logscale', 'when to store results to pickle files '
                                                    '[yes|no|logscale|best|never|final]'),
        BeamParam('store_networks', str, 'best/last', 'when to store network weights to the log directory '
                                         '[yes|no|logscale|best|all_bests|never|final|last|best/last]'),

        BeamParam('comet', bool, False, 'Whether to use comet.ml for logging'),
        BeamParam('git_directory', str, None, 'The git directory to use for comet.ml logging'),
        BeamParam('comet_workspace', str, None, 'The comet.ml workspace to use for logging'),

        BeamParam('mlflow_url', str, None, 'The url of the mlflow serve to use for logging. '
                                           'If None, mlflow will log to $MLFLOW_TRACKING_URI'),

        BeamParam('training_framework', str, 'torch', 'Chose between [torch|amp|accelerate|deepspeed]'),
        BeamParam('compile_train', bool, False,
                  'Apply torch.compile to optimize the inner_train function to speed up training. '
                  'To use this feature, you must override and use the alg.inner_train function '
                  'in your alg.train_iteration function'),
        BeamParam('compile_network', bool, False,
                  'Apply torch.compile to optimize the network forward function to speed up training.'),

        BeamParam('objective', str, 'objective', 'A single objective to apply hyperparameter optimization or '
                                                 'ReduceLROnPlateau scheduling. '
                                                 'By default we consider maximization of the objective (e.g. accuracy) '
                                                 'You can override this behavior by overriding the Algorithm.report method.'),

        BeamParam('optimization_mode', str, None, 'Set [min/max] to minimize/maximize the objective. '
                                               'By default objectives that contain the words "loss/error/mse" a are minimized and '
                                               'other objectives are maximized. You can override this behavior by setting this flag.'),

        BeamParam('objective_to_report', str, 'best', 'Which objective to report in HPO run [best|last]'),

        # possible combinations for single gpu:
        # 1. torch
        # 2. amp
        # 3. accelerate
        # 4. native deepspeed

        # possible combinations for multiple gpus:
        # 1. torch + ddp
        # 2. amp + ddp
        # 3. accelerate + deepspeed
        # 4. native deepspeed
    ]


class NNExperimentConfig(ExperimentConfig, NNTrainingConfig, DDPConfig):
    pass


class TransformerConfig(CacheConfig):
    # transformer arguments

    parameters = [
        BeamParam('mp_method', str, 'joblib', 'The multiprocessing method to use'),
        BeamParam('n_chunks', int, None, 'The number of chunks to split the dataset'),
        BeamParam('name', str, None, 'The name of the dataset'),
        BeamParam('store_path', str, None, 'The path to store the results'),
        BeamParam('partition', str, None, 'The partition to use for splitting the dataset'),
        BeamParam('chunksize', int, None, 'The chunksize to use for splitting the dataset'),
        BeamParam('squeeze', bool, False, 'Whether to squeeze the chunks (e.g. 1-dim dataframe to series)'),
        BeamParam('reduce', bool, True, 'Whether to reduce and collate the results'),
        BeamParam('reduce_dim', int, 0, 'The dimension to reduce the results'),
        BeamParam('transform_strategy', str, None, 'The transform strategy to use can be [CC|CS|SC|SS]'),
        BeamParam('store_chunk', bool, None, 'Whether to store the chunked results [None stores chunks if '
                                             'n_chunks/chunksize is not None and store_path is not None]'),
        BeamParam('split_by', str, 'keys', 'The split strategy to use can be [keys|index|columns]'),
        BeamParam('store_suffix', str, None, 'The suffix to add to the stored file'),
        BeamParam('override', bool, False, 'Whether to override the stored file if it exists'),
        BeamParam('use-dill', bool, False, 'Whether to use dill for serialization'),
        BeamParam('return-results', bool, None, 'Whether to return the results if None, it is set to True '
                                                'if store_path is None'),
        BeamParam('n_workers', int, None, 'The number of workers to use for the transformation. '
                                          'If None defaults to 1 if chunksize is not None and n_chunks otherwise,'
                                          'if <1 defaults to half of the number of cpus'),
        BeamParam('use-cache', bool, False, 'Use the store_path as cache and do not apply transformation '
                                            'if cache exists'),
        BeamParam('retries', int, 1, 'The number of retries to apply for each chunk'),
        BeamParam('retries_delay', float, 1., 'The delay between retries'),
        BeamParam('strict-transform', bool, False, 'Whether to raise an exception if the transformation fails'),

    ]


class UniversalConfig(NNExperimentConfig, TransformerConfig):
    pass


def beam_arguments(*args, return_defaults=False, return_tags=False, **kwargs):

    if len(args) and type(args[0]) == argparse.ArgumentParser:
        pr = args[0]
        args = args[1:]
    else:
        pr = empty_beam_parser()

    args = [pr] + list(args)
    return _beam_arguments(*args, return_defaults=return_defaults, return_tags=return_tags, **kwargs)
