import argparse
import copy
import os
import sys
from argparse import Namespace
import re
from collections import defaultdict

from ..utils import is_notebook, check_type
from ..path import beam_path, beam_key
from ..type import Types
from ..logging import beam_logger as logger
from .._version import __version__
import re


def boolean_feature(parser, feature, default=False, help='', metavar=None):

    if type(feature) is not str:
        featurename = feature[0].replace("-", "_")
    else:
        featurename = feature.replace("-", "_")
        feature = [feature]

    feature_parser = parser.add_mutually_exclusive_group(required=False)

    for f in feature:
        f_dash = f.replace("_", "-")
        f_under = f.replace("-", "_")

        feature_parser.add_argument(f"--{f_dash}", f"--{f_under}", dest=featurename, action='store_true',
                                    help=help)
        feature_parser.add_argument(f"--no-{f_dash}", f"--no-{f_under}",  dest=featurename, action='store_false',
                                    help=help)

    pa = parser._actions
    for a in pa:
        if a.dest == featurename:
            a.metavar = metavar
    parser.set_defaults(**{featurename: default})


def empty_beam_parser():

    parser = argparse.ArgumentParser(description='List of available arguments for this project',
                                     conflict_handler='resolve')
    parser.add_argument('config_files', nargs='*',
                        help='A list config files (optional) to load, will be loaded in reverse order '
                             '(first one has higher priority). Pass explicit arguments to override these configs')
    return parser


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


# Function to compare parsed args with defaults
def args_that_were_provided_explicitly(parser, args: dict = None, sys_argv=True):
    # Collect all argument strings (flags) for each action
    action_to_flags = {}
    for action in parser._actions:
        if action.option_strings:
            action_to_flags[action.dest] = action.option_strings

    # Check which of these were actually used in sys.argv
    passed_actions = set()
    if sys_argv:
        for arg in sys.argv[1:]:  # skip the first element, which is the script name
            for dest, flags in action_to_flags.items():
                if arg.split('=')[0] in flags:
                    passed_actions.add(dest)
                    break
    if args is not None:
        for k in args.keys():
            # add k to the set of passed actions if it was explicitly set to a non-default value
            if k in action_to_flags and args[k] != parser.get_default(k):
                passed_actions.add(k)

    return passed_actions


def add_unknown_arguments(args, unknown, silent=False):
    args = copy.deepcopy(args)

    i = 0

    if len(unknown) > 0 and not silent:
        logger.warning(f"Parsing unknown arguments: {unknown}. Please check for typos")

    while i < len(unknown):

        arg = unknown[i]
        if not arg.startswith("-"):
            if not silent:
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
                if not silent:
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


def _beam_arguments(*args, return_defaults=False, return_tags=False, silent=False,
                    strict=False, load_config_files=True, load_script_arguments=True, **kwargs):
    '''
    args can be list of arguments or a long string of arguments or list of strings each contains multiple arguments
    kwargs is a dictionary of both defined and undefined arguments
    priority order: script arguments > non default kwargs+args > config files > environment variable > default values
    '''

    sys_argv_copy = sys.argv.copy()

    pr = args[0]
    args = args[1:]

    org_pr = copy.deepcopy(pr)
    def update_parser(p, d):
        for pi in p._actions:
            for o in pi.option_strings:
                o = o.replace('--', '').replace('-', '_')
                if o in d:
                    p.set_defaults(**{pi.dest: d[o]})

    if '---BEAM-JUPYTER---' in sys.argv:
        file_name = sys.argv[0]
        sys.argv = sys.argv[sys.argv.index('---BEAM-JUPYTER---') + 1:]
        sys.argv.insert(0, file_name)

    elif is_notebook() or not load_script_arguments:
        sys.argv = sys.argv[:1]

    file_name = sys.argv[0] if len(sys.argv) > 0 else '/tmp/tmp.py'
    sys_args = sys.argv[1:]

    args_str = []
    args_dict = []

    config_files = []

    for ar in args:

        ar_type = check_type(ar)

        if isinstance(ar, Namespace):
            args_dict.append(to_dict(ar))
        elif ar_type.minor == Types.dict:
            args_dict.append(ar)
        elif ar_type.minor == Types.path:
            config_files.append(ar)
        elif ar_type.major == Types.scalar and ar_type.element == Types.str:
            args_str.append(ar)
        else:
            raise ValueError

    for ar in args_dict:
        kwargs = {**ar, **kwargs}

    args_str = re.split(r"\s+", ' '.join([ar.strip() for ar in args_str]))

    sys.argv = [file_name] + args_str + sys_args
    sys.argv = list(filter(lambda x: bool(x), sys.argv))

    update_parser(pr, kwargs)
    # set defaults from environment variables
    update_parser(pr, os.environ)

    config_files_should_be_loaded = False
    if return_defaults:
        args = pr.parse_args([])
        keys_with_non_default_values = set()

    else:
        args, unknown = pr.parse_known_args()

        if load_config_files:
            if hasattr(args, 'config_files') and args.config_files:
                config_files_should_be_loaded = True
            elif len(config_files):
                config_files_should_be_loaded = True

        if config_files_should_be_loaded:
            keys_with_non_default_values = set(args_that_were_provided_explicitly(pr, sys_argv=True))
        else:
            keys_with_non_default_values = set()

        if not strict:
            args_with_unknowns = add_unknown_arguments(args, unknown, silent=silent)
            if config_files_should_be_loaded:
                keys_with_non_default_values.update(set(vars(args_with_unknowns).keys()).difference(vars(args).keys()))
            args = args_with_unknowns

    for k, v in kwargs.items():
        if k.startswith('_'):
            continue
        if k not in args:
            setattr(args, k, v)

    tags = defaultdict(set)
    if config_files_should_be_loaded:
        config_files.extend(args.config_files)
        delattr(args, 'config_files')

        config_args = {}

        for config_file in config_files:
            try:
                cf = beam_path(config_file).read()
            except FileNotFoundError as e:
                logger.error(f"Config file {config_file} not found. make sure you follow the pattern --arg=value "
                             f"(--arg value would lead to errors like this). If you want to load a config file, "
                             f"make sure it exists in the path.")
                raise e

            if '_tags' in cf:
                for k, v in cf['_tags'].items():
                    tags[k].add(v)
                del cf['_tags']

            cf = {c.replace('-', '_'): v for c, v in cf.items()}
            config_args.update(cf)

        # remove from config_args all the keys with non-default values s.t. the arguments passed in the command line
        # have higher priority

        for k in keys_with_non_default_values:
            logger.debug(f"loading {k} from command line instead of config files {config_files}. "
                         f"command line arguments have higher priority over config files")
            config_args.pop(k, None)

        kwargs_non_default_values = set(args_that_were_provided_explicitly(org_pr, kwargs, sys_argv=False))
        for k in kwargs_non_default_values:
            logger.debug(f"loading {k} from kwargs instead of config files {config_files}. "
                         f"non default kwargs has higher priority over config files")
            config_args.pop(k, None)

        args = Namespace(**{**to_dict(args), **config_args})
    elif hasattr(args, 'config_files'):
        delattr(args, 'config_files')

    beam_key.set_hparams(to_dict(args))

    if not return_tags:
        return args

    for pai in pr._actions:
        tag_list = get_tags_from_action(pai)
        for tag in tag_list:
            tags[tag].add(pai.dest)

    sys.argv = sys_argv_copy

    return args, tags


def get_tags_from_action(action):
    tags = set()
    if action.metavar is not None:
        tags = re.findall(r'[a-zA-Z0-9_-]+', action.metavar)
    return list(tags)


def get_beam_llm(llm_uri=None, get_from_key=True):
    llm = None
    if llm_uri is None and get_from_key:
        llm_uri = beam_key('BEAM_LLM', store=False)
    if llm_uri is not None:
        try:
            from ..llm import beam_llm
            llm = beam_llm(llm_uri)
        except ImportError:
            pass
    return llm


def print_beam_hyperparameters(args, default_params=None, debug_only=False):
    if debug_only:
        log_func = logger.debug
    else:
        log_func = logger.info

    log_func(f"Beam experiment (Beam version: {__version__})")
    log_func(f"project: {args.project_name}, algorithm: {args.algorithm}, identifier: {args.identifier}")
    log_func(f"Global paths:")
    log_func(f"data-path (where the dataset should be): {args.data_path}")
    log_func(f"logs-path (where the experiments are log to): {args.logs_path}")
    log_func(f'Experiment objective: {args.objective} (set for schedulers, early stopping and best checkpoint store)')
    log_func('Experiment Hyperparameters (only non default values are listed):')
    log_func('----------------------------------------------------------'
             '---------------------------------------------------------------------')

    if hasattr(args, 'hparams'):
        hparams_list = args.hparams
    else:
        hparams_list = args

    var_args_sorted = dict(sorted(to_dict(args).items()))

    for k, v in var_args_sorted.items():
        if k == 'hparams':
            continue
        elif (default_params is not None and k in hparams_list and hasattr(default_params, k) and
              (v is not None and v != getattr(default_params, k))):
            log_func(k + ': ' + str(v))
        else:
            logger.debug(k + ': ' + str(v))

    log_func('----------------------------------------------------------'
             '---------------------------------------------------------------------')
