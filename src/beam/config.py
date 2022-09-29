import argparse
import copy
import os
import sys
from .utils import is_notebook, check_type, logger
import re
import math


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


def get_beam_parser():

    # add a general argument parser, arguments may be overloaded
    parser = argparse.ArgumentParser(description='List of available arguments for this project', conflict_handler='resolve')
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
    parser.add_argument('--project-name', type=str, default='beam', help='The name of the beam project')
    parser.add_argument('--algorithm', type=str, default='Algorithm', help='algorithm name')
    parser.add_argument('--identifier', type=str, default='debug', help='The name of the model to use')

    parser.add_argument('--mp-port', type=str, default='random', help='Port to be used for multiprocessing')

    parser.add_argument('--root-dir', type=str, default=os.path.join(os.path.expanduser('~'), 'beam_projects'),
                         help='Root directory for Logs and results')

    boolean_feature(parser, "reload", False, "Load saved model")
    parser.add_argument('--resume', type=int, default=-1,
                        help='Resume experiment number, set -1 for last experiment: active when reload=True')
    boolean_feature(parser, "override", False, "Override last experiment: active when reload=False")

    parser.add_argument('--cpu-workers', type=int, default=0, help='How many CPUs will be used for the data loading')
    parser.add_argument('--device', type=str, default='0', help='GPU Number or cpu/cuda string')
    parser.add_argument("--device-list", nargs="+", default=None,
                        help='Set GPU priority for parallel execution e.g. --device-list 2 1 3 will use GPUs 2 and 1 '
                        'when passing --parallel=2 and will use GPUs 2 1 3 when passing --parallel=3. '
                        'If None, will use an ascending order starting from the GPU passed in the --device parameter.'
                        'e.g. when --device=1 will use GPUs 1,2,3,4 when --parallel=4')

    parser.add_argument('--parallel', type=int, default=1, metavar='hparam',
                        help='Number of parallel gpu workers. Set <=1 for single process')
    parser.add_argument('--schedulers-steps', type=str, default='epoch',
                        help='When to apply schedulers steps [epoch|iteration|none]: each epoch or each iteration '
                             'Use none to avoid scheduler steps or to use your own custom steps policy')
    parser.add_argument('--scheduler', type=str, default=None, help='Build BeamScheduler. Supported schedulers: '
                                                                    '[one_cycle,reduce_on_plateau,cosine_annealing]')
    parser.add_argument('--objective', type=str, default=None,
                        help='A single objective to apply hyperparameter optimization or ReduceLROnPlateau scheduling. '
                             'By default we consider maximization of the objective (e.g. accuracy) '
                             'You can override this behavior by overriding the Algorithm.report method.')

    # booleans

    boolean_feature(parser, "tensorboard", True, "Log results to tensorboard")
    boolean_feature(parser, "lognet", True, 'Log  networks parameters')
    boolean_feature(parser, "deterministic", False, 'Use deterministic pytorch optimization for reproducability'
                                                    'when enabling non-deterministic behavior, it sets '
                                                    'torch.backends.cudnn.benchmark = True which'
                                                    'accelerates the computation')
    boolean_feature(parser, "scale-epoch-by-batch-size", True,
                    'When True: epoch length corresponds to the number of examples sampled from the dataset in each epoch'
                    'When False: epoch length corresponds to the number of forward passes in each epoch')

    boolean_feature(parser, "half", False, "Use FP16 instead of FP32", metavar='hparam')
    boolean_feature(parser, "amp", False, "Use Automatic Mixed Precision", metavar='hparam')
    boolean_feature(parser, "scalene", False, "Profile the experiment with the Scalene python profiler")

    boolean_feature(parser, "find-unused-parameters", False, "For DDP applications: allows running backward on "
                                                             "a subgraph of the model. introduces extra overheads, "
                                                             "so applications should only set find_unused_parameters "
                                                             "to True when necessary")
    boolean_feature(parser, "broadcast-buffers", True, "For DDP applications: Flag that enables syncing (broadcasting) "
                                                       "buffers of the module at beginning of the forward function.")

    boolean_feature(parser, "store-initial-weights", False, "Store the network's initial weights")
    boolean_feature(parser, "capturable", False, 'Temporary workaround that should be removed in future pytorch releases '
                                                 'it makes possible to reload models with adam optimizers '
                                                 'see: https://github.com/pytorch/pytorch/issues/80809')

    # experiment parameters
    parser.add_argument('--init', type=str, default='ortho', metavar='hparam',
                        help='Initialization method [ortho|N02|xavier|]')
    parser.add_argument('--seed', type=int, default=0, help='Seed for reproducability (zero is saved for random seed)')
    parser.add_argument('--split-dataset-seed', type=int, default=5782,
                        help='Seed dataset split (set to zero to get random split)')

    parser.add_argument('--total-steps', type=int, default=int(1e6), metavar='hparam', help='Total number of environment steps')

    parser.add_argument('--epoch-length', type=int, default=None, metavar='hparam',
                        help='Length of train+eval epochs (if None - it is taken from epoch-length-train/epoch-length-eval arguments)')
    parser.add_argument('--epoch-length-train', type=int, default=None, metavar='hparam',
                        help='Length of each epoch (if None - it is the dataset[train] size)')
    parser.add_argument('--epoch-length-eval', type=int, default=None, metavar='hparam',
                        help='Length of each evaluation epoch (if None - it is the dataset[validation] size)')
    parser.add_argument('--n-epochs', type=int, default=None, metavar='hparam',
                        help='Number of epochs, if None, it uses the total steps to determine the number of iterations')

    boolean_feature(parser, "dynamic-sampler", False, 'Whether to use a dynamic sampler (mainly for rl/optimization)')
    parser.add_argument('--buffer-size', type=int, default=None, metavar='hparam',
                        help='Maximal Dataset size in dynamic problems')
    parser.add_argument('--probs-normalization', type=str, default='sum',
                        help='Sampler\'s probabilities normalization method [sum/softmax]')
    parser.add_argument('--sample-size', type=int, default=100000, help='Periodic sample size for the dynamic sampler')
    # environment parameters

    # Learning parameters

    parser.add_argument('--batch-size', type=int, default=256, metavar='hparam', help='Batch Size')
    parser.add_argument('--batch-size-train', type=int, default=None, metavar='hparam', help='Batch Size for training iterations')
    parser.add_argument('--batch-size-eval', type=int, default=None, metavar='hparam', help='Batch Size for testing/evaluation iterations')

    parser.add_argument('--reduction', type=str, metavar='hparam', default='sum', help='whether to sum loss elements or average them')
    parser.add_argument('--lr-dense', '--lr', type=float, default=1e-3, metavar='hparam', help='learning rate for dense optimizers')
    parser.add_argument('--lr-sparse', type=float, default=1e-2, metavar='hparam', help='learning rate for sparse optimizers')
    parser.add_argument('--cycle-max-momentum', type=float, default=.95, metavar='hparam',
                        help='The maximum momentum in one-cycle optimizer')
    parser.add_argument('--cycle-base-momentum', type=float, default=.85, metavar='hparam',
                        help='The base momentum in one-cycle optimizer')
    parser.add_argument('--cawr-t0', type=int, default=10, metavar='hparam',
                        help=' Number of iterations for the first restart in CosineAnnealingWarmRestarts scheduler')
    parser.add_argument('--cawr-tmult', type=int, default=1, metavar='hparam',
                        help=' A factor increases Ti after a restart in CosineAnnealingWarmRestarts scheduler')
    parser.add_argument('--scheduler-factor', type=float, default=math.sqrt(.1), metavar='hparam',
                        help='The factor to reduce lr in schedulers such as ReduceOnPlateau')
    parser.add_argument('--scheduler-patience', type=int, default=None, metavar='hparam',
                        help='Patience for the ReduceOnPlateau scheduler')
    parser.add_argument('--scheduler-warmup', type=float, default=5, metavar='hparam',
                        help='Scheduler\'s warmup factor (in epochs)')
    parser.add_argument('--weight-decay', type=float, default=0., metavar='hparam', help='L2 regularization coefficient for dense optimizers')
    parser.add_argument('--eps', type=float, default=1e-4, metavar='hparam', help='Adam\'s epsilon parameter')
    parser.add_argument('--momentum', '--beta1', type=float, default=0.9, metavar='hparam',
                        help='The momentum and Adam\'s β1 parameter')
    parser.add_argument('--beta2', type=float, default=0.999, metavar='hparam', help='Adam\'s β2 parameter')
    parser.add_argument('--clip-gradient', type=float, default=0., metavar='hparam', help='Clip Gradient L2 norm')
    parser.add_argument('--accumulate', type=int, default=1, metavar='hparam', help='Accumulate gradients for this number of backward iterations')
    parser.add_argument('--oversampling-factor', type=float, default=.0, metavar='hparam',
                        help='A factor [0, 1] that controls how much to oversample where'
                             '0-no oversampling and 1-full oversampling. Set 0 for no oversampling')
    parser.add_argument('--expansion-size', type=int, default=int(1e7),
                        help='largest expanded index size for oversampling')

    parser.add_argument('--swa', type=float, default=None,
                        help='SWA period. If float it is a fraction of the total number of epochs. '
                             'If integer, it is the number of SWA epochs.')
    parser.add_argument('--swa-lr', type=float, default=0.05, metavar='hparam', help='The SWA learning rate')
    parser.add_argument('--swa-anneal-epochs', type=int, default=10, metavar='hparam', help='The SWA lr annealing period')

    # results printing and visualization

    boolean_feature(parser, "print-results", True, "Print results after each epoch to screen")
    boolean_feature(parser, "visualize-weights", True, "Visualize network weights on tensorboard")
    boolean_feature(parser, "enable-tqdm", True, "Print tqdm progress bar when training")
    parser.add_argument('--tqdm-threshold', type=float, default=10., help='Minimal expected epoch time to print tqdm bar'
                                                                         'set 0 to ignore and determine tqdm bar with tqdm-enable flag')
    parser.add_argument('--tqdm-stats', type=float, default=1., help='Take this period to calculate the experted epoch time')

    parser.add_argument('--visualize-results', type=str, default='yes',
                        help='when to visualize results on tensorboard [yes|no|logscale]')
    parser.add_argument('--store-results', type=str, default='logscale',
                        help='when to store results to pickle files')
    parser.add_argument('--store-networks', type=str, default='logscale',
                        help='when to store network weights to the log directory')

    return parser


def normalize_key(k):
    return k.replace('-', '_')

def normalize_value(v):
    try:
        return int(v)
    except:
        pass
    try:
        return float(v)
    except:
        pass
    return v


def add_unknown_arguments(args, unknown):

    args = copy.deepcopy(args)

    i = 0

    if len(unknown) > 0:
        logger.warning(f"Parsing unkown arguments: {unknown}. Please check for typos")

    while i < len(unknown):

        arg = unknown[i]
        if not arg.startswith("-"):
            logger.error(f"Cannot correctly parse: {unknown[i]} arguments as it as it does not start with \'-\' sign")
            i += 1
            continue
        if arg.startswith("--"):
            arg = arg[2:]
        else:
            arg = arg[1:]

        if arg.startswith('no-'):
            k = arg[3:]
            setattr(args, normalize_key(k), False)
            i += 1
            continue

        if '=' in arg:
            arg = arg.split('=')
            if len(arg) != 2:
                logger.error(f"Cannot correctly parse: {unknown[i]} arguments as it contains more than one \'=\' sign")
                i += 1
                continue
            k, v = arg
            setattr(args, normalize_key(k), normalize_value(v))
            i += 1
            continue

        k = normalize_key(arg)
        if i == len(unknown) - 1 or unknown[i+1].startswith("-"):
            setattr(args, k, True)
            i += 1
        else:
            v = unknown[i+1]
            setattr(args, k, normalize_value(v))
            i += 2

    return args


def beam_arguments(*args, **kwargs):
    '''
    args can be list of arguments or a long string of arguments or list of strings each contains multiple arguments
    kwargs is a dictionary of both defined and undefined arguments
    '''

    if is_notebook():
        sys.argv = sys.argv[:1]

    file_name = sys.argv[0] if len(sys.argv) > 0 else '/tmp/tmp.py'
    sys_args = sys.argv[1:]

    args_str = []
    args_dict = []

    if len(args) and type(args[0]) == argparse.ArgumentParser:
        pr = args[0]
        args = args[1:]
    else:
        pr = get_beam_parser()

    for ar in args:

        ar_type = check_type(ar)

        if ar_type.major == 'dict':
            args_dict.append(ar)
        elif ar_type.major == 'scalar' and ar_type.element == 'str':
            args_str.append(ar)
        else:
            raise ValueError

    args_str = re.split(r"\s+", ' '.join([ar.strip() for ar in args_str]))

    sys.argv = [file_name] + args_str + sys_args
    sys.argv = list(filter(lambda x: bool(x), sys.argv))

    args, unknown = pr.parse_known_args()
    args = add_unknown_arguments(args, unknown)

    for k, v in kwargs.items():
        setattr(args, k, v)

    for ar in args_dict:
        for k, v in ar.items():
            setattr(args, k, v)

    hparams = [pai.dest for pai in pr._actions if pai.metavar == 'hparam']
    setattr(args, 'hparams', hparams)

    return args
