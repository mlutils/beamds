import argparse
import os
import math
from pathlib import Path


def boolean_feature(parser, feature, default=False, help='', metavar=None):
    featurename = feature.replace("-", "_")
    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--%s' % feature, dest=featurename, action='store_true', help=help)
    feature_parser.add_argument('--no-%s' % feature, dest=featurename, action='store_false', help=help)
    pa = parser._actions
    for a in pa:
        if a.dest == featurename:
            a.metavar = metavar
    parser.set_defaults(**{featurename: default})


def empty_beam_parser():

    parser = argparse.ArgumentParser(description='List of available arguments for this project',
                                     conflict_handler='resolve')
    return parser


def basic_beam_parser():
    # add a general argument parser, arguments may be overloaded
    parser = argparse.ArgumentParser(description='List of available arguments for this project',
                                     conflict_handler='resolve')
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

    parser.add_argument('experiment_configuration', nargs='?', default=None,
                        help='A config file (optional) for the current experiment. '
                             'If not provided no config file will be loaded')
    parser.add_argument('--project-name', type=str, default='beam', help='The name of the beam project')
    parser.add_argument('--algorithm', type=str, default='Algorithm', help='algorithm name')
    parser.add_argument('--identifier', type=str, default='debug', help='The name of the model to use')

    parser.add_argument('--mp-port', type=str, default='random', help='Port to be used for multiprocessing')
    parser.add_argument('--beam-llm', type=str, default=None, help='URI of the LLM service')

    parser.add_argument('--logs-path', type=str,
                        default=os.path.join(os.path.expanduser('~'), 'beam_projects', 'experiments'),
                        help='Root directory for Logs and results')

    parser.add_argument('--data-path', type=str,
                        default=os.path.join(os.path.expanduser('~'), 'beam_projects', 'data'),
                        help='Where the dataset is located')

    boolean_feature(parser, "reload", False, "Load saved model")
    parser.add_argument('--resume', type=int, default=-1,
                        help='Resume experiment number, set -1 for last experiment: active when reload=True')
    boolean_feature(parser, "override", False, "Override last experiment: active when reload=False")
    parser.add_argument('--reload-checkpoint', type=str, default='best',
                        help='Which checkpoint to reload [best|last|<epoch>]')

    parser.add_argument('--cpu-workers', type=int, default=0, help='How many CPUs will be used for the data loading')
    parser.add_argument('--data-fetch-timeout', type=float, default=0., help='Timeout for the dataloader fetching. '
                                                                             'set to 0 for no timeout.')
    parser.add_argument('--device', type=str, default='0', help='GPU Number or cpu/cuda string')
    parser.add_argument("--device-list", nargs="+", default=None,
                        help='Set GPU priority for parallel execution e.g. --device-list 2 1 3 will use GPUs 2 and 1 '
                             'when passing --n-gpus=2 and will use GPUs 2 1 3 when passing --n-gpus=3. '
                             'If None, will use an ascending order starting from the GPU passed in the --device parameter.'
                             'e.g. when --device=1 will use GPUs 1,2,3,4 when --n-gpus=4')

    parser.add_argument('--n-gpus', type=int, default=1, metavar='tune',
                        help='Number of parallel gpu workers. Set <=1 for single process')
    parser.add_argument('--schedulers-steps', type=str, default='epoch', metavar='tune',
                        help='When to apply schedulers steps [epoch|iteration|none]: each epoch or each iteration '
                             'Use none to avoid scheduler steps or to use your own custom steps policy')
    parser.add_argument('--scheduler', type=str, default=None, metavar='tune',
                        help='Build BeamScheduler. Supported schedulers: '
                             '[one_cycle,reduce_on_plateau,cosine_annealing]')
    parser.add_argument('--objective', type=str, default='objective',
                        help='A single objective to apply hyperparameter optimization or ReduceLROnPlateau scheduling. '
                             'By default we consider maximization of the objective (e.g. accuracy) '
                             'You can override this behavior by overriding the Algorithm.report method.')

    parser.add_argument('--objective-mode', type=str, default=None,
                        help='Set [min/max] to minimize/maximize the objective.'
                             'By default objectives that contain the words "loss/error/mse" a are minimized and '
                             'other objectives are maximized. You can override this behavior by setting this flag.')

    # booleans

    boolean_feature(parser, "tensorboard", True, "Log results to tensorboard")
    boolean_feature(parser, "mlflow", False, "Log results to MLFLOW serve")

    boolean_feature(parser, "lognet", True, 'Log  networks parameters')
    boolean_feature(parser, "deterministic", False, 'Use deterministic pytorch optimization for reproducability'
                                                    'when enabling non-deterministic behavior, it sets '
                                                    'torch.backends.cudnn.benchmark = True which'
                                                    'accelerates the computation')
    boolean_feature(parser, "scale-epoch-by-batch-size", True,
                    'When True: epoch length corresponds to the number of examples sampled from the dataset in each epoch'
                    'When False: epoch length corresponds to the number of forward passes in each epoch')

    parser.add_argument('--model-dtype', type=str, default='float32', metavar='tune/model',
                        help='dtype, both for automatic mixed precision and accelerate. Supported dtypes: '
                             '[float32, float16, bfloat16]')
    boolean_feature(parser, "scalene", False, "Profile the experiment with the Scalene python profiler")
    boolean_feature(parser, "safetensors", False, "Save tensors in safetensors format instead of "
                                                  "native torch")
    boolean_feature(parser, "find-unused-parameters", False, "For DDP applications: allows running backward on "
                                                             "a subgraph of the model. introduces extra overheads, "
                                                             "so applications should only set find_unused_parameters "
                                                             "to True when necessary")
    boolean_feature(parser, "broadcast-buffers", True, "For DDP applications: Flag that enables syncing (broadcasting) "
                                                       "buffers of the module at beginning of the forward function.")

    boolean_feature(parser, "store-initial-weights", False, "Store the network's initial weights")
    boolean_feature(parser, "capturable", False,
                    'Temporary workaround that should be removed in future pytorch releases '
                    'it makes possible to reload models with adam optimizers '
                    'see: https://github.com/pytorch/pytorch/issues/80809')
    boolean_feature(parser, "copy-code", True, "Copy the code directory into the experiment directory")
    boolean_feature(parser, "restart-epochs-count", True,
                    "When reloading an algorithm, restart counting epochs from zero "
                    "(with respect to schedulers and swa training)", metavar='tune')

    # experiment parameters
    parser.add_argument('--init', type=str, default='ortho', metavar='tune',
                        help='Initialization method [ortho|N02|xavier|]')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for reproducability (zero is saved for random seed)')
    parser.add_argument('--split-dataset-seed', type=int, default=5782,
                        help='Seed dataset split (set to zero to get random split)')
    parser.add_argument('--test-size', type=float, default=.2, help='Test set percentage')
    parser.add_argument('--validation-size', type=float, default=.2, help='Validation set percentage')

    parser.add_argument('--total-steps', type=int, default=int(1e6), metavar='tune',
                        help='Total number of environment steps')

    parser.add_argument('--epoch-length', type=int, default=None, metavar='tune',
                        help='Length of train+eval epochs (if None - it is taken from epoch-length-train/epoch-length-eval arguments)')
    parser.add_argument('--epoch-length-train', type=int, default=None, metavar='tune',
                        help='Length of each epoch (if None - it is the dataset[train] size)')
    parser.add_argument('--epoch-length-eval', type=int, default=None, metavar='tune',
                        help='Length of each evaluation epoch (if None - it is the dataset[validation] size)')
    parser.add_argument('--n-epochs', type=int, default=None, metavar='tune',
                        help='Number of epochs, if None, it uses the total steps to determine the number of iterations')

    boolean_feature(parser, "dynamic-sampler", False,
                    'Whether to use a dynamic sampler (mainly for rl/optimization)')
    parser.add_argument('--buffer-size', type=int, default=None, metavar='tune',
                        help='Maximal Dataset size in dynamic problems')
    parser.add_argument('--probs-normalization', type=str, default='sum',
                        help='Sampler\'s probabilities normalization method [sum/softmax]')
    parser.add_argument('--sample-size', type=int, default=100000,
                        help='Periodic sample size for the dynamic sampler')
    # environment parameters

    # Learning parameters

    parser.add_argument('--train-timeout', type=int, default=None, metavar='tune',
                        help='Timeout for the training in seconds. Set to None for no timeout')
    parser.add_argument('--batch-size', type=int, default=256, metavar='tune', help='Batch Size')
    parser.add_argument('--batch-size-train', type=int, default=None, metavar='tune',
                        help='Batch Size for training iterations')
    parser.add_argument('--batch-size-eval', type=int, default=None, metavar='tune',
                        help='Batch Size for testing/evaluation iterations')

    parser.add_argument('--reduction', type=str, metavar='tune', default='sum',
                        help='whether to sum loss elements or average them [sum|mean|mean_batch|sqrt|mean_sqrt]')
    parser.add_argument('--lr-dense', '--lr', type=float, default=1e-3, metavar='tune',
                        help='learning rate for dense optimizers')
    parser.add_argument('--lr-sparse', type=float, default=1e-2, metavar='tune',
                        help='learning rate for sparse optimizers')
    parser.add_argument('--cycle-max-momentum', type=float, default=.95, metavar='tune',
                        help='The maximum momentum in one-cycle optimizer')
    parser.add_argument('--cycle-base-momentum', type=float, default=.85, metavar='tune',
                        help='The base momentum in one-cycle optimizer')
    parser.add_argument('--cawr-t0', type=int, default=10, metavar='tune',
                        help=' Number of iterations for the first restart in CosineAnnealingWarmRestarts scheduler')
    parser.add_argument('--cawr-tmult', type=int, default=1, metavar='tune',
                        help=' A factor increases Ti after a restart in CosineAnnealingWarmRestarts scheduler')
    parser.add_argument('--scheduler-factor', '--scheduler-gamma', type=float,
                        default=math.sqrt(.1), metavar='tune',
                        help='The factor to reduce lr in schedulers such as ReduceOnPlateau')
    parser.add_argument('--scheduler-patience', type=int, default=None, metavar='tune',
                        help='Patience for the ReduceOnPlateau scheduler')
    parser.add_argument('--scheduler-warmup', type=float, default=5, metavar='tune',
                        help='Scheduler\'s warmup factor (in epochs)')
    parser.add_argument('--weight-decay', type=float, default=0., metavar='tune',
                        help='L2 regularization coefficient for dense optimizers')
    parser.add_argument('--eps', type=float, default=1e-4, metavar='tune', help='Adam\'s epsilon parameter')
    parser.add_argument('--momentum', '--beta1', type=float, default=0.9, metavar='tune',
                        help='The momentum and Adam\'s β1 parameter')
    parser.add_argument('--beta2', type=float, default=0.999, metavar='tune', help='Adam\'s β2 parameter')
    parser.add_argument('--clip-gradient', type=float, default=0., metavar='tune', help='Clip Gradient L2 norm')
    parser.add_argument('--accumulate', type=int, default=1, metavar='tune',
                        help='Accumulate gradients for this number of backward iterations')
    parser.add_argument('--oversampling-factor', type=float, default=.0, metavar='tune',
                        help='A factor [0, 1] that controls how much to oversample where'
                             '0-no oversampling and 1-full oversampling. Set 0 for no oversampling')
    parser.add_argument('--expansion-size', type=int, default=int(1e7),
                        help='largest expanded index size for oversampling')
    parser.add_argument('--stop-at', type=float, default=0., metavar='tune',
                        help='Early stopping when objective >= stop_at')
    parser.add_argument('--early-stopping-patience', type=int, default=0, metavar='tune',
                        help='Early stopping patience in epochs, '
                             'stop when current_epoch - best_epoch >= early_stopping_patience')

    parser.add_argument('--swa', type=float, default=None,
                        help='SWA period. If float it is a fraction of the total number of epochs. '
                             'If integer, it is the number of SWA epochs.')
    parser.add_argument('--swa-lr', type=float, default=0.05, metavar='tune', help='The SWA learning rate')
    parser.add_argument('--swa-anneal-epochs', type=int, default=10, metavar='tune', help='The SWA lr annealing period')

    # results printing and visualization

    boolean_feature(parser, "print-results", True, "Print results after each epoch to screen")
    boolean_feature(parser, "visualize-weights", True, "Visualize network weights on tensorboard")
    boolean_feature(parser, "enable-tqdm", True, "Print tqdm progress bar when training")
    parser.add_argument('--visualize-results-log-base', type=int, default=10,
                        help='log base for the logarithmic based results visualization')
    parser.add_argument('--tqdm-threshold', type=float, default=10.,
                        help='Minimal expected epoch time to print tqdm bar'
                             'set 0 to ignore and determine tqdm bar with tqdm-enable flag')
    parser.add_argument('--tqdm-stats', type=float, default=1.,
                        help='Take this period to calculate the experted epoch time')

    parser.add_argument('--visualize-results', type=str, default='yes',
                        help='when to visualize results on tensorboard [yes|no|logscale]')
    parser.add_argument('--store-results', type=str, default='logscale',
                        help='when to store results to pickle files')
    parser.add_argument('--store-networks', type=str, default='logscale',
                        help='when to store network weights to the log directory')

    parser.add_argument('--mp-context', type=str, default='spawn', help='The multiprocessing context to use')
    parser.add_argument('--mp-backend', type=str, default=None, help='The multiprocessing backend to use')

    boolean_feature(parser, "comet", False, "Whether to use comet.ml for logging")
    parser.add_argument('--git-directory', type=str, default=None, help='The git directory to use for comet.ml logging')
    parser.add_argument('--comet-workspace', type=str, default=None, help='The comet.ml workspace to use for logging')

    parser.add_argument('--config-file', type=str, default=str(Path.home().joinpath('conf.pkl')),
                        help='The beam config file to use with secret keys')

    parser.add_argument('--mlflow-url', type=str, default=None, help='The url of the mlflow serve to use for logging. '
                                                                     'If None, mlflow will log to $MLFLOW_TRACKING_URI')
    # keys
    parser.add_argument('--comet-api-key', type=str, default=None, help='The comet.ml api key to use for logging')
    parser.add_argument('--aws-access-key', type=str, default=None, help='The aws access key to use for S3 connections')
    parser.add_argument('--aws-private-key', type=str, default=None,
                        help='The aws private key to use for S3 connections')
    parser.add_argument('--ssh-secret-key', type=str, default=None,
                        help='The ssh secret key to use for ssh connections')
    parser.add_argument('--openai-api-key', type=str, default=None,
                        help='The openai api key to use for openai connections')

    # catboost

    boolean_feature(parser, "cb-ranker", False,
                    "Whether to use catboost ranker instead of regression", metavar='model')
    parser.add_argument('--cb-n-estimators', type=int, default=1000, metavar='tune',
                        help='The number of trees in the catboost model')

    # transformer arguments
    parser.add_argument('--mp-method', type=str, default='joblib', help='The multiprocessing method to use')
    parser.add_argument('--n-chunks', type=int, default=None, metavar='tune',
                        help='The number of chunks to split the dataset')
    parser.add_argument('--name', type=str, default=None, metavar='tune',
                        help='The name of the dataset')
    parser.add_argument('--store-path', type=str, default=None, help='The path to store the results')
    parser.add_argument('--partition', type=str, default=None, help='The partition to use for splitting the dataset')
    parser.add_argument('--chunksize', type=int, default=None, help='The chunksize to use for splitting the dataset')
    parser.add_argument('--squeeze', type=bool, default=True, help='Whether to squeeze the results')
    parser.add_argument('--reduce', type=bool, default=True, help='Whether to reduce and collate the results')
    parser.add_argument('--reduce-dim', type=int, default=0, help='The dimension to reduce the results')
    parser.add_argument('--transform-strategy', type=str, default=None,
                        help='The transform strategy to use can be [CC|CS|SC|SS]')
    parser.add_argument('--split-by', type=str, default='keys',
                        help='The split strategy to use can be [keys|index|columns]')
    parser.add_argument('--store-suffix', type=str, default=None, help='The suffix to add to the stored file')

    parser.add_argument('--llm', type=str, default=None, metavar='model',
                        help='URI of a Large Language Model to be used in the experiment.')

    # accelerate parameters
    # based on https://huggingface.co/docs/accelerate/v0.24.0/en/package_reference/accelerator#accelerate.Accelerator

    # boolean_feature(parser, "deepspeed-dataloader", False,
    #                 "Use optimized deepspeed dataloader instead of native pytorch dataloader")
    boolean_feature(parser, "device-placement", False,
                    " Whether or not the accelerator should put objects on device")
    boolean_feature(parser, "split-batches", False,
                    "Whether or not the accelerator should split the batches "
                    "yielded by the dataloaders across the devices")

    parser.add_argument('--deepspeed-optimizer', type=str, default='AdamW',
                        help='Optimizer type (currently used for deepspeed configuration only) '
                             'Supported optimizers: [Adam, AdamW, Lamb, OneBitAdam, OneBitLamb]')

    parser.add_argument('--deepspeed-config', type=str, default=None,
                        help='Path to a deepspeed configuration file.')
    parser.add_argument('--zero-stage', type=int, default=2, help='The ZeRO training stage to use.')
    parser.add_argument('--distributed-backend', type=str, default=None,
                        help='The distributed backend to use. Supported backends: [nccl, gloo, mpi]')

    # boolean_feature(parser, "accelerate", False, "Whether to use accelerate package for training")
    # boolean_feature(parser, "half", False, "Use FP16 instead of FP32", metavar='tune/model')
    # boolean_feature(parser, "amp", False, "Use Automatic Mixed Precision", metavar='tune/model')

    # parser.add_argument('--parallel-backend', type=str, default='ddp',
    #                     help='Chose between [ddp|deepspeed|horovord], currently only ddp is supported')

    parser.add_argument('--training-framework', type=str, default='torch',
                        help='Chose between [torch|amp|accelerate|deepspeed]')

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

    boolean_feature(parser, "compile-train", False,
                    "Apply torch.compile to optimize the inner_train function to speed up training. "
                    "To use this feature, you must override and use the alg.inner_train function "
                    "in your alg.train_iteration function")

    return parser
