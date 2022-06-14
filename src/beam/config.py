import argparse
import os
import sys
from .utils import is_notebook


def boolean_feature(feature, default, help):
    global parser
    featurename = feature.replace("-", "_")
    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--%s' % feature, dest=featurename, action='store_true', help=help)
    feature_parser.add_argument('--no-%s' % feature, dest=featurename, action='store_false', help=help)
    parser.set_defaults(**{featurename: default})


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
parser.add_argument('--identifier', type=str, default='debug', help='The name of the model to use')
parser.add_argument('--algorithm', type=str, default='Algorithm', help='algorithm name')
parser.add_argument('--mp-port', type=str, default=None, help='Port to be used for multiprocessing')

parser.add_argument('--root-dir', type=str, default=os.path.join(os.path.expanduser('~'), 'beam_projects'),
                    help='Root directory for Logs and results')

boolean_feature("reload", False, "Load saved model")
parser.add_argument('--resume', type=int, default=-1,
                    help='Resume experiment number, set -1 for last experiment: active when reload=True')
boolean_feature("override", False, "Override last experiment: active when reload=False")

parser.add_argument('--cpu-workers', type=int, default=0, help='How many CPUs will be used for the data loading')
parser.add_argument('--device', type=str, default='0', help='GPU Number or cpu/cuda string')
parser.add_argument('--parallel', type=int, default=1,
                    help='Number of parallel gpu workers. Set <=1 for single process')

# booleans

boolean_feature("tensorboard", True, "Log results to tensorboard")
boolean_feature("lognet", True, 'Log  networks parameters')
boolean_feature("deterministic", False, 'Use deterministic pytorch optimization for reproducability')
boolean_feature("scale-epoch-by-batch-size", True,
                'When True: epoch length corresponds to the number of examples sampled from the dataset in each epoch'
                'When False: epoch length corresponds to the number of forward passes in each epoch')

boolean_feature("cudnn-benchmark", False, "Setting torch.backends.cudnn.benchmark = True before the training loop can accelerate the computation")
boolean_feature("half", False, "Use FP16 instead of FP32")
boolean_feature("amp", False, "Use Automatic Mixed Precision")

# experiment parameters
parser.add_argument('--init', type=str, default='ortho', help='Initialization method [ortho|N02|xavier|]')
parser.add_argument('--seed', type=int, default=0, help='Seed for reproducability (zero is saved for random seed)')

parser.add_argument('--total-steps', type=int, default=int(1e6), help='Total number of environment steps')

parser.add_argument('--epoch-length', type=int, default=None,
                    help='Length of both train/eval epochs (if None - it is taken from epoch-length-train/epoch-length-eval arguments)')
parser.add_argument('--epoch-length-train', type=int, default=None,
                    help='Length of each epoch (if None - it is the dataset[train] size)')
parser.add_argument('--epoch-length-eval', type=int, default=None,
                    help='Length of each evaluation epoch (if None - it is the dataset[validation] size)')
parser.add_argument('--n-epochs', type=int, default=None,
                    help='Number of epochs, if None, it uses the total steps to determine the number of iterations')

# environment parameters

# Learning parameters

parser.add_argument('--batch-size', type=int, default=256, help='Batch Size')
parser.add_argument('--batch-size-train', type=int, default=None, help='Batch Size for training iterations')
parser.add_argument('--batch-size-eval', type=int, default=None, help='Batch Size for testing/evaluation iterations')

parser.add_argument('--lr-dense', type=float, default=1e-3, metavar='α', help='learning rate for dense optimizers')
parser.add_argument('--lr-sparse', type=float, default=1e-2, metavar='α', help='learning rate for sparse optimizers')
parser.add_argument('--weight-decay', type=float, default=0., help='L2 regularization coefficient for dense optimizers')
parser.add_argument('--eps', type=float, default=1e-4, metavar='ɛ', help='Adam\'s epsilon parameter')
parser.add_argument('--beta1', type=float, default=0.9, metavar='β', help='Adam\'s β1 parameter')
parser.add_argument('--beta2', type=float, default=0.999, metavar='β', help='Adam\'s β2 parameter')
parser.add_argument('--clip', type=float, default=0., help='Clip Pi Gradient L2 norm')
parser.add_argument('--accumulate', type=int, default=1, help='Accumulate gradients for this number of backward iterations')

# results printing and visualization

boolean_feature("print-results", True, "Print results after each epoch to screen")
boolean_feature("visualize-weights", True, "Visualize network weights on tensorboard")
boolean_feature("enable-tqdm", True, "Print tqdm progress bar when training")

parser.add_argument('--visualize-results', type=str, default='yes',
                    help='when to visualize results on tensorboard [yes|no|logscale]')
parser.add_argument('--store-results', type=str, default='logscale', help='when to store results to pickle files')
parser.add_argument('--store-networks', type=str, default='logscale',
                    help='when to store network weights to the log directory')


def beam_arguments(*args, **kwargs):
    '''
    args can be list of arguments or a long string of arguments or list of strings each contains multiple arguments
    kwargs is a dictionary of both defined and undefined arguments
    '''

    if is_notebook():
        sys.argv = sys.argv[:1]

    file_name = sys.argv[0] if len(sys.argv) > 0 else '/tmp/tmp.py'

    args = ' '.join(args).split(' ')
    sys.argv = [file_name] + args

    args = parser.parse_args()

    for k, v in kwargs.items():
        setattr(args, k, v)

    return args
