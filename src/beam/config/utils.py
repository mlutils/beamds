import argparse
import copy
import os
import sys
from argparse import Namespace
import re

from .basic_configuration import basic_beam_parser
from ..utils import is_notebook, check_type
from ..path import beam_path, beam_key
from ..logger import beam_logger as logger
from .._version import __version__


def to_dict(hparams):
    if hasattr(hparams, 'items'):
        return dict(hparams.items())
    return vars(hparams)


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
        if i == len(unknown) - 1 or unknown[i + 1].startswith("-"):
            setattr(args, k, True)
            i += 1
        else:
            v = unknown[i + 1]
            setattr(args, k, normalize_value(v))
            i += 2

    return args


def beam_arguments(*args, return_defaults=False, **kwargs):
    '''
    args can be list of arguments or a long string of arguments or list of strings each contains multiple arguments
    kwargs is a dictionary of both defined and undefined arguments
    '''

    def update_parser(p, d):
        for pi in p._actions:
            for o in pi.option_strings:
                o = o.replace('--', '').replace('-', '_')
                if o in d:
                    p.set_defaults(**{pi.dest: d[o]})

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
        pr = basic_beam_parser()

    for ar in args:

        ar_type = check_type(ar)

        if isinstance(ar, Namespace):
            args_dict.append(to_dict(ar))
        elif ar_type.minor == 'dict':
            args_dict.append(ar)
        elif ar_type.major == 'scalar' and ar_type.element == 'str':
            args_str.append(ar)
        else:
            raise ValueError

    for ar in args_dict:
        kwargs = {**kwargs, **ar}

    args_str = re.split(r"\s+", ' '.join([ar.strip() for ar in args_str]))

    sys.argv = [file_name] + args_str + sys_args
    sys.argv = list(filter(lambda x: bool(x), sys.argv))

    update_parser(pr, kwargs)
    # set defaults from environment variables
    update_parser(pr, os.environ)

    if return_defaults:
        args = pr.parse_args([])
    else:
        args, unknown = pr.parse_known_args()
        args = add_unknown_arguments(args, unknown)

    for k, v in kwargs.items():
        if k not in args:
            setattr(args, k, v)

    if args.experiment_configuration is not None:
        cf = beam_path(args.experiment_configuration).read()
        for k, v in cf.items():
            setattr(args, k, v)

    beam_key.set_hparams(to_dict(args))

    tune = [pai.dest for pai in pr._actions if pai.metavar is not None and 'tune' in pai.metavar]
    setattr(args, 'tune_set', set(tune))

    model = [pai.dest for pai in pr._actions if pai.metavar is not None and 'model' in pai.metavar]
    setattr(args, 'model_set', set(model))

    return args


def get_beam_llm(llm_uri=None, get_from_key=True):
    llm = None
    if llm_uri is None and get_from_key:
        llm_uri = beam_key('BEAM_LLM', store=False)
    if llm_uri is not None:
        try:
            from .llm import beam_llm
            llm = beam_llm(llm_uri)
        except ImportError:
            pass
    return llm


def print_beam_hyperparameters(args, debug_only=False):
    if debug_only:
        log_func = logger.debug
    else:
        log_func = logger.info

    log_func(f"Beam experiment (Beam version: {__version__})")
    log_func(f"project: {args.project_name}, algorithm: {args.algorithm}, identifier: {args.identifier}")
    log_func(f"Global paths:")
    log_func(f"path-to-data (where the dataset should be): {args.path_to_data}")
    log_func(f"root-dir (where results are written to): {args.root_dir}")
    log_func(f'Experiment objective: {args.objective} (set for schedulers, early stopping and best checkpoint store)')
    log_func('Experiment Hyperparameters (only non default values are listed):')
    log_func('----------------------------------------------------------'
             '---------------------------------------------------------------------')

    if hasattr(args, 'hparams'):
        hparams_list = args.hparams
    else:
        hparams_list = args

    var_args_sorted = dict(sorted(to_dict(args).items()))

    default_params = basic_beam_parser()

    for k, v in var_args_sorted.items():
        if k == 'hparams':
            continue
        elif k in hparams_list and (v is not None and v != default_params.get_default(k)):
            log_func(k + ': ' + str(v))
        else:
            logger.debug(k + ': ' + str(v))

    log_func('----------------------------------------------------------'
             '---------------------------------------------------------------------')
