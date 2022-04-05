import argparse


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
parser.add_argument('--identifier', type=str, default='debug', help='The name of the model to use')
parser.add_argument('--algorithm', type=str, default='Algorithm', help='algorithm name')
boolean_feature("reload", False, "Load saved model")
parser.add_argument('--resume', type=int, default=-1, help='Resume experiment number, set -1 for last experiment: active when reload=True')
boolean_feature("override", True, "Override last experiment: active when reload=False")


parser.add_argument('--cpu-workers', type=int, default=0, help='How many CPUs will be used for the data loading')
parser.add_argument('--cuda', type=int, default=0, help='GPU Number, set -1 for cpu only')
parser.add_argument('--parallel', type=int, default=1, help='Number of parallel gpu workers. Set <=1 for single process')

# booleans

boolean_feature("tensorboard", True, "Log results to tensorboard")
boolean_feature("lognet", True, 'Log  networks parameters')
boolean_feature("deterministic", False, 'Use deterministic pytorch optimization for reproducability')

# experiment parameters
parser.add_argument('--init', type=str, default='ortho', help='Initialization method [ortho|N02|xavier|]')
parser.add_argument('--seed', type=int, default=0, help='Seed for reproducability (zero is saved for random seed)')

parser.add_argument('--total-steps', type=int, default=int(1e6), help='Total number of environment steps')
parser.add_argument('--epoch-length', type=int, default=1000, help='Length of each epoch')

# environment parameters

# Netwoks parameters

parser.add_argument('--batch-size', type=int, default=256, help='Batch Size')

parser.add_argument('--lr-d', type=float, default=1e-3, metavar='α', help='learning rate for dense optimizers')
parser.add_argument('--lr-s', type=float, default=1e-2, metavar='α', help='learning rate for sparse optimizers')
parser.add_argument('--weight-decay', type=float, default=0., help='L2 regularization coefficient for dense optimizers')

parser.add_argument('--dropout', type=float, default=0., help='Dropout regularization coefficient')
parser.add_argument('--clip-p', type=float, default=0., help='Clip Pi Gradient L2 norm')
parser.add_argument('--clip-q', type=float, default=0., help='Clip Pi Gradient L2 norm')
parser.add_argument('--weight-decay-p', type=float, default=0, help='Weight decay coefficient for Q-net')
parser.add_argument('--weight-decay-q', type=float, default=0, help='Weight decay coefficient for π-net')
parser.add_argument('--min-std', type=int, default=1e-3, help='min log')
parser.add_argument('--max-std', type=int, default=1e3, help='max log')
