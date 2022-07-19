import re
import sys
import time
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from shutil import copytree
import torch
import copy
import shutil
from collections import defaultdict
from .utils import include_patterns, logger, check_type, beam_device, check_element_type
from .algorithm import Algorithm
import pandas as pd
import torch.multiprocessing as mp
from .utils import setup, cleanup, set_seed, find_free_port, check_if_port_is_available, is_notebook
import torch.distributed as dist
from functools import partial
from argparse import Namespace
from tensorboard.notebook import start as start_tensorboard

done = mp.Event()


def beam_algorithm_generator(experiment, Alg, Dataset=None):

    if issubclass(type(Alg), Algorithm):
        alg = Alg
    else:
        alg = Alg(experiment.hparams)
        # if a new algorithm is generated, we clean the tensorboard writer. If the reload option is True,
        # the algorithm will fix the epoch number s.t. tensorboard graphs will not overlap
        experiment.writer_cleanup()

    if issubclass(type(Dataset), torch.utils.data.Dataset):
        dataset = Dataset
    elif Dataset is None:
        dataset = alg.dataset
    else:
        dataset = Dataset(experiment.hparams)

    alg.load_dataset(dataset)
    alg.experiment = experiment

    return alg


def default_runner(rank, world_size, experiment, algorithm_generator, *args, tensorboard_arguments=None, **kwargs):
    alg = algorithm_generator(*args, **kwargs)

    experiment.writer_control(enable=not (bool(rank)))
    results = {}

    try:
        for i, results in enumerate(iter(alg)):
            experiment.save_model_results(results, alg, i,
                                          print_results=experiment.hparams.print_results,
                                          visualize_results=experiment.hparams.visualize_results,
                                          store_results=experiment.hparams.store_results, store_networks=experiment.hparams.store_networks,
                                          visualize_weights=experiment.hparams.visualize_weights,
                                          argv=tensorboard_arguments)

    except KeyboardInterrupt:

        logger.error(f"KeyboardInterrupt: Training was interrupted, Worker terminates")
        if world_size == 1:
            logger.error(f"KeyboardInterrupt: Training was interrupted, reloads last checkpoint")
            experiment.reload_checkpoint(alg)

    if world_size == 1:
        return alg, results


def run_worker(rank, world_size, results_queue, job, experiment, *args, **kwargs):

    logger.info(f"Worker: {rank + 1}/{world_size} is running...")

    if world_size > 1:
        setup(rank, world_size, port=experiment.hparams.mp_port)

    experiment.set_rank(rank, world_size)
    set_seed(seed=experiment.hparams.seed, constant=rank+1, increment=False, deterministic=experiment.hparams.deterministic)

    res = job(rank, world_size, experiment, *args, **kwargs)

    if world_size > 1:

        cleanup(rank, world_size)
        results_queue.put({'rank': rank, 'results': res})

        done.wait()

    else:
        return res


class Experiment(object):
    """
    Experiment name:
    <algorithm name>_<identifier>_exp_<number>_<time>


    Experiment number and overriding experiments

    These parameters are responsible for which experiment to load or to generate:
    the name of the experiment is <alg>_<identifier>_exp_<num>_<time>
    The possible configurations:
    reload = False, override = True: always overrides last experiment (default configuration)
    reload = False, override = False: always append experiment to the list (increment experiment num)
    reload = True, resume = -1: resume to the last experiment
    reload = True, resume = <n>: resume to the <n> experiment


    :param args:
    """

    def __init__(self, args, results_names=None, hpo=None, trial=None, print_hyperparameters=True):
        """
        args: the parsed arguments
        results_names: additional results directories (defaults are: train, validation, test)
        """

        self.tensorboard_hparams = {}

        pa = None
        if hasattr(args, 'parser'):
            pa = args.parser._actions
            pa = {pai.dest: pai.metavar for pai in pa}
            delattr(args, 'parser')

        vars_args = copy.copy(vars(args))
        for k, v in vars_args.items():
            param_type = check_type(v)
            if param_type.element in ['bool', 'str', 'int', 'float'] and pa is not None and k in pa and pa[k] == 'hparam':
                self.tensorboard_hparams[k] =v

        self.hparams = copy.copy(args)

        set_seed(seed=self.hparams.seed, constant=0, increment=False, deterministic=self.hparams.deterministic)

        # parameters
        self.start_time = time.time()
        self.exptime = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self.hparams.device = beam_device(self.hparams.device)

        self.base_dir = os.path.join(self.hparams.root_dir, self.hparams.project_name,
                                     self.hparams.algorithm, self.hparams.identifier)
        os.makedirs(self.base_dir, exist_ok=True)

        self.exp_name = None
        self.load_model = False

        pattern = re.compile("\A\d{4}_\d{8}_\d{6}\Z")
        exp_names = list(filter(lambda x: re.match(pattern, x) is not None, os.listdir(self.base_dir)))
        exp_indices = np.array([int(d.split('_')[0]) for d in exp_names])

        if self.hparams.reload:

            if type(self.hparams.resume) is str:
                if os.path.isdir(os.path.join(self.base_dir, self.hparams.resume)):
                    self.exp_name = self.hparams.resume
                    exp_num = int(self.exp_name.split('_')[0])
                    self.load_model = True

            elif self.hparams.resume >= 0:
                ind = np.nonzero(exp_indices == self.hparams.resume)[0]
                if len(ind):
                    self.exp_name = exp_names[ind[0]]
                    exp_num = self.hparams.resume
                    self.load_model = True

            else:
                if len(exp_indices):
                    ind = np.argmax(exp_indices)
                    self.exp_name = exp_names[ind]
                    exp_num = exp_indices[ind]
                    self.load_model = True

        else:

            if self.hparams.override and len(exp_indices):

                ind = np.argmax(exp_indices)
                self.exp_name = exp_names[ind]
                exp_num = exp_indices[ind]
            else:
                self.hparams.override = False

        if self.exp_name is None:
            exp_num = np.max(exp_indices) + 1 if len(exp_indices) else 0
            self.exp_name = "%04d_%s" % (exp_num, self.exptime)

        # init experiment parameters
        self.root = os.path.join(self.base_dir, self.exp_name)

        # set dirs
        self.tensorboard_dir = os.path.join(self.root, 'tensorboard')
        self.checkpoints_dir = os.path.join(self.root, 'checkpoints')
        self.results_dir = os.path.join(self.root, 'results')
        self.code_dir = os.path.join(self.root, 'code')

        if self.load_model:
            logger.info("Resuming existing experiment")

        else:

            if not self.hparams.override:
                logger.info("Creating new experiment")

            else:
                logger.info("Deleting old experiment")

                shutil.rmtree(self.root)
                self.exp_name = "%04d_%s" % (exp_num, self.exptime)
                self.root = os.path.join(self.base_dir, self.exp_name)

                # set dirs
                self.tensorboard_dir = os.path.join(self.root, 'tensorboard')
                self.checkpoints_dir = os.path.join(self.root, 'checkpoints')
                self.results_dir = os.path.join(self.root, 'results')
                self.code_dir = os.path.join(self.root, 'code')

            logger.info(f"Experiment directory is: {self.root}")

            os.makedirs(os.path.join(self.tensorboard_dir, 'logs'))
            os.makedirs(os.path.join(self.tensorboard_dir, 'hparams'))
            os.makedirs(self.checkpoints_dir)

            # make log dirs
            os.makedirs(os.path.join(self.results_dir, 'train'))
            os.makedirs(os.path.join(self.results_dir, 'validation'))
            os.makedirs(os.path.join(self.results_dir, 'test'))

            if type(results_names) is list:
                for r in results_names:
                    os.makedirs(os.path.join(self.results_dir, r))
            elif type(results_names) is str:
                os.makedirs(os.path.join(self.results_dir, results_names))

            # copy code to dir

            if is_notebook():
                code_root_path = os.getcwd()
            else:
                code_root_path = sys.argv[0]

            copytree(os.path.dirname(os.path.realpath(code_root_path)), self.code_dir,
                     ignore=include_patterns('*.py', '*.md', '*.ipynb'))

            pd.to_pickle(vars_args, os.path.join(self.root, "args.pkl"))

        self.writer = None
        self.rank = 0
        self.world_size = args.parallel

        if self.world_size > 1:
            torch.multiprocessing.set_sharing_strategy('file_system')

        log_file = os.path.join(self.root, "experiment.log")
        logger.add(log_file, level='INFO', colorize=True,
                   format='<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>')

        if print_hyperparameters:
            logger.info(f"beam project: {args.project_name}")
            logger.info('Experiment Hyperparameters')

            for k, v in vars_args.items():
                logger.info(k + ': ' + str(v))

        # replace zero split_dataset_seed to none (non-deterministic split) - if zero
        if self.hparams.split_dataset_seed == 0:
            self.hparams.split_dataset_seed = None

        # fill the batch size

        if self.hparams.batch_size_train is None:
            self.hparams.batch_size_train = self.hparams.batch_size

        if self.hparams.batch_size_eval is None:
            self.hparams.batch_size_eval = self.hparams.batch_size

        if self.hparams.batch_size is None:
            self.hparams.batch_size = self.hparams.batch_size_train

        if self.hparams.epoch_length_train is None:
            self.hparams.epoch_length_train = self.hparams.epoch_length

        if self.hparams.epoch_length_eval is None:
            self.hparams.epoch_length_eval = self.hparams.epoch_length

        # build the hyperparamter class which will be sent to the dataset and algorithm classes

        if self.load_model:
            self.hparams.reload_path = self.reload_checkpoint()
        else:
            self.hparams.reload_path = None

        self.trial = trial
        self.hparams.hpo = hpo
        self.hparams.ddp = False

        pd.to_pickle(self.tensorboard_hparams, os.path.join(self.root, "hparams.pkl"))

    @staticmethod
    def reload_from_path(path, **argv):

        logger.info(f"Reload experiment from path: {path}")
        args = pd.read_pickle(os.path.join(path, "args.pkl"))
        args = Namespace(**args)
        args.override = False
        args.reload = True

        path, d = os.path.split(path)
        if not d:
            path, d = os.path.split(path)
        args.resume = d

        return Experiment(args, **argv)

    def reload_checkpoint(self, alg=None, iloc=-1, loc=None, name=None):

        checkpoints = os.listdir(self.checkpoints_dir)
        if not(len(checkpoints)):
            logger.error(f"Directory of checkpoints is empty")
            return

        checkpoints = pd.DataFrame({'name': checkpoints, 'index': [int(c.split('_')[-1]) for c in checkpoints]})
        checkpoints = checkpoints.sort_values('index')

        if name is not None:
            path = os.path.join(self.checkpoints_dir, name)
        elif loc is not None:
            chp = checkpoints.loc[loc]['name']
            path = os.path.join(self.checkpoints_dir, chp)
        else:
            chp = checkpoints.iloc[iloc]['name']
            path = os.path.join(self.checkpoints_dir, chp)

        logger.info(f"Reload experiment from checkpoint: {path}")

        if alg is not None:
            alg.load_checkpoint(path)

        else:
            return path

    def set_rank(self, rank, world_size):

        self.rank = rank
        self.world_size = world_size
        self.hparams.ddp = self.world_size > 1
        self.hparams.enable_tqdm = self.hparams.enable_tqdm and (rank == 0)

        if 'cpu' not in str(self.hparams.device) and world_size > 1:
            self.hparams.device = beam_device(rank)

    def writer_control(self, enable=True, networks=None, inputs=None):

        if enable and self.writer is None and self.hparams.tensorboard:
            self.writer = SummaryWriter(log_dir=os.path.join(self.tensorboard_dir, 'logs'),
                                        comment=self.hparams.identifier)

        if networks is not None and enable and self.writer is not None:
            for k, net in networks.items():
                self.writer.add_graph(net, inputs[k])

    def save_model_results(self, results, algorithm, iteration, visualize_results='yes',
                           store_results='logscale', store_networks='logscale', print_results=True,
                           visualize_weights=False, argv=None):

        '''

        responsible for 4 actions:
        1. print results to stdout
        2. visualize results via tensorboard
        3. store results to pandas pickle objects
        4. save networks and optimizers

        logscale is active only on integer epochs in logscale (see x-axis in plt.semilogx)

        :param results:
        :param algorithm:
        :param visualize_results: takes yes|no|logscale.
        :param store_results: takes yes|no|logscale.
        :param store_networks: takes yes|no|logscale.
        :param print_results: whether to print the results to stdout when saving results to tensorboard.
        :return:
        '''

        epoch = algorithm.epoch
        if not self.rank:

            if print_results:
                logger.info('')
                logger.info(f'Finished epoch {iteration+1}/{algorithm.n_epochs} (Total trained epochs {epoch}).')

            decade = int(np.log10(epoch) + 1)
            logscale = not (epoch - 1) % (10 ** (decade - 1))

            for subset, res in results.items():

                if store_results == 'yes' or store_results == 'logscale' and logscale:
                    pd.to_pickle(res, os.path.join(self.results_dir, subset, f'results_{epoch:06d}'))

                alg = algorithm if visualize_weights else None

            if visualize_results == 'yes' or visualize_results == 'logscale' and logscale:
                self.log_data(copy.deepcopy(results), epoch, print_log=print_results, alg=alg, argv=argv)

            checkpoint_file = os.path.join(self.checkpoints_dir, f'checkpoint_{epoch:06d}')
            algorithm.save_checkpoint(checkpoint_file)

            if store_networks == 'no' or store_networks == 'logscale' and not logscale:
                try:
                    os.remove(os.path.join(self.checkpoints_dir, f'checkpoint_{epoch - 1:06d}'))
                except OSError:
                    pass

        if self.world_size > 1:
            dist.barrier()

    def log_data(self, results, n, print_log=True, alg=None, argv=None):

        for subset, res in results.items():

            def format(v):
                v_type = check_element_type(v)
                if v_type == 'int':
                    if v >= 1000:
                        return f"{float(v): .4}"
                    else:
                        return str(v)
                elif v_type == 'float':
                    return f"{v: .4}"
                else:
                    return v

            logger.info(f'{subset}:')

            if print_log and 'stats' in res:
                logger.info('| '.join([f"{k}: {format(v)} " for k, v in res['stats'].items()]))

            report = None
            if issubclass(type(res), dict):
                if 'scalar' in res:
                    report = res['scalar']
                else:
                    report = res

            if report is not None:

                for param, val in report.items():

                    stat = None

                    if type(val) is dict or type(val) is defaultdict:
                        for p, v in val.items():
                            val[p] = np.mean(v)
                    elif isinstance(report[param], torch.Tensor):
                        stat = pd.Series(report[param].cpu().numpy()).describe()
                        report[param] = torch.mean(val)
                    else:
                        stat = pd.Series(report[param]).describe()
                        report[param] = np.mean(val)

                    if print_log:
                        if not (type(report[param]) is dict or type(
                                report[param]) is defaultdict):
                            stat = '| '.join([f"{k if k != 'mean' else 'avg'}:{v: .4}".ljust(15) for k, v in dict(stat).items() if k != 'count'])
                            paramp = f'{param}:'
                            logger.info(f'{paramp: <12} | {stat}')

        if self.writer is None:
            return

        defaults_argv = defaultdict(lambda: defaultdict(dict))
        if argv is not None:
            for log_type in argv:
                for k in argv[log_type]:
                    defaults_argv[log_type][k] = argv[log_type][k]

        if alg is not None:
            networks = alg.get_networks()
            for net in networks:
                for name, param in networks[net].named_parameters():
                    try:
                        self.writer.add_histogram("weight_%s/%s" % (net, name), param.data.cpu().numpy(), n,
                                                  bins='tensorflow')
                        self.writer.add_histogram("grad_%s/%s" % (net, name), param.grad.cpu().numpy(), n,
                                                  bins='tensorflow')
                        if hasattr(param, 'intermediate'):
                            self.writer.add_histogram("iterm_%s/%s" % (net, name), param.intermediate.cpu().numpy(),
                                                      n,
                                                      bins='tensorflow')
                    except:
                        pass
        metrics = {}
        for subset, res in results.items():
            if issubclass(type(res), dict) and subset != 'objective':
                for log_type in res:
                    if hasattr(self.writer, f'add_{log_type}'):
                        log_func = getattr(self.writer, f'add_{log_type}')
                        for param in res[log_type]:
                            if type(res[log_type][param]) is dict or type(res[log_type][param]) is defaultdict:
                                for p, v in res[log_type][param].items():
                                    log_func(f'{subset}_{param}/{p}', v, n, **defaults_argv[log_type][param])
                            elif type(res[log_type][param]) is list:
                                log_func(f'{subset}/{param}', *res[log_type][param], n, **defaults_argv[log_type][param])
                            else:
                                log_func(f'{subset}/{param}', res[log_type][param], n, **defaults_argv[log_type][param])
                                if log_type == 'scalar':
                                    metrics[f"{subset}/{param}"] = float(res[log_type][param])

        if len(metrics):
            self.writer.add_hparams(self.tensorboard_hparams, metrics, name=os.path.join('..', 'hparams'), global_step=n)
            # self.writer.add_hparams(self.tensorboard_hparams, metrics, run_name=os.path.join('..', 'hparams'))

    def tensorboard(self, port=None, add_all_of_same_identifier=False, add_all_of_same_algorithm=False,
                          add_all_of_same_project=False, more_experiments=None, more_identifiers=None,
                          more_algorithms=None, get_port_from_beam_port_range=True, hparams=False):

        suffix = 'hparams' if hparams else 'logs'

        if port is None:
            if get_port_from_beam_port_range:
                base_range = int(os.environ['JUPYTER_PORT']) // 1000
                port_range = range(base_range * 1000, (base_range + 1) * 1000)

            else:
                port_range = range(10000, 2**16)

            for p in port_range:
                if check_if_port_is_available(p):
                    port = str(p)
                    break

            if port is None:
                logger.error("Cannot find free port in the specified range")
                return

        else:
            if not check_if_port_is_available(port):
                logger.error(f"Port {port} is not available")
                return

        logger.info(f"Opening a tensorboard server on port: {port}")

        def gen_hparams_string(e):
            tensorboard_hparams = pd.read_pickle(os.path.join(e, "hparams.pkl"))
            return '/'.join([f"{k}_{v}" for k, v in tensorboard_hparams.items()])

        def path_depth(path):
            return len(os.path.normpath(path).split(os.sep))

        def normalize_path(path, level=0):

            normal_path = [self.hparams.root_dir, self.hparams.project_name,
                           self.hparams.algorithm, self.hparams.identifier]
            pd = path_depth(self.hparams.root_dir)

            return os.path.join(*normal_path[:len(normal_path)-pd-level], path)

        if add_all_of_same_project:
            base_dir = os.path.join(self.hparams.root_dir, self.hparams.project_name)
            depth = 3
        elif add_all_of_same_algorithm:
            base_dir = os.path.join(self.hparams.root_dir, self.hparams.project_name, self.hparams.algorithm)
            depth = 2
        elif add_all_of_same_identifier:
            base_dir = os.path.join(self.hparams.root_dir, self.hparams.project_name, self.hparams.algorithm, self.hparams.identifier)
            depth = 1
        else:
            base_dir = self.root
            depth = 0

        experiments = [d[0] for d in list(os.walk(base_dir)) if (path_depth(d[0]) - path_depth(base_dir)) == depth]

        if more_experiments is not None:
            if hparams:
                logger.error("hparams visualization does not support adding additional experiments")
            if type(more_experiments) is str:
                more_experiments = [more_experiments]
                experiments = experiments + [normalize_path(e, level=0) for e in more_experiments]

        if more_identifiers is not None:
            if hparams:
                logger.error("hparams visualization does not support adding additional experiments")
            if type(more_identifiers) is str:
                more_identifiers = [more_identifiers]
                depth = 1
                for identifier in more_identifiers:
                    identifier = normalize_path(identifier, level=depth)
                    experiments = experiments + [d[0] for d in list(os.walk(identifier)) if (path_depth(d[0]) - path_depth(identifier)) == depth]

        if more_algorithms is not None:
            if hparams:
                logger.error("hparams visualization does not support adding additional experiments")
            if type(more_algorithms) is str:
                more_algorithms = [more_algorithms]
                depth = 2
                for algorithm in more_algorithms:
                    algorithm = normalize_path(algorithm, level=depth)
                    experiments = experiments + [d[0] for d in list(os.walk(algorithm)) if (path_depth(d[0]) - path_depth(algorithm)) == depth]

        experiments = [os.path.normpath(e) for e in experiments]
        names = ['/'.join(e.split(os.sep)[-3:]) for e in experiments]
        names = [f"{n}/{gen_hparams_string(e)}" for n, e in zip(names, experiments)]

        experiments = [os.path.join(e, 'tensorboard', suffix) for e in experiments]
        log_dirs = ','.join([f"{n}:{e}" for n, e in zip(names, experiments)])

        if hparams:
            command_argument = f"--bind_all --logdir {base_dir} --port {port}"
        else:
            command_argument = f"--bind_all --logdir_spec={log_dirs} --port {port}"
        start_tensorboard(command_argument)

    def algorithm_generator(self, Alg, Dataset=None):
        return beam_algorithm_generator(self, Alg=Alg, Dataset=Dataset)

    def fit(self, Alg, Dataset=None, *args, return_results=False, reload_results=False,
            tensorboard_arguments=None, **kwargs):

        ag = partial(beam_algorithm_generator, Alg=Alg, Dataset=Dataset)
        return self(ag, *args, return_results=return_results, reload_results=reload_results,
                    tensorboard_arguments=tensorboard_arguments, **kwargs)

    def __call__(self, algorithm_generator, *args, return_results=False, reload_results=False,
                  tensorboard_arguments=None, **kwargs):

        try:
            res = self.run(default_runner, *(algorithm_generator, self, *args),
                           tensorboard_arguments=tensorboard_arguments, **kwargs)

        except KeyboardInterrupt:

            res = None
            logger.error(f"KeyboardInterrupt: Training was interrupted, reloads last checkpoint")

        if res is None or self.world_size > 1:
            alg = algorithm_generator(self, *args, **kwargs)
            self.reload_checkpoint(alg)

            if reload_results:
                results = {}
                for subset in os.listdir(alg.results_dir):

                    res = os.listdir(os.path.join(alg.results_dir, subset))
                    res = pd.DataFrame({'name': res, 'index': [int(c.split('_')[-1]) for c in res]})
                    res = res.sort_values('index')

                    res = res.iloc['name']
                    path = os.path.join(alg.results_dir, subset, res)

                    results[subset] = path

                if reload_results:
                    results = {subset: pd.read_pickle(path) for subset, path in results.items()}

        else:
            alg, results = res

        if return_results:
            return alg, results
        else:
            return alg

    def run(self, job, *args, **kwargs):

        arguments = (job, self, *args)

        def _run(demo_fn, world_size):

            ctx = mp.get_context('spawn')
            results_queue = ctx.Queue()
            for rank in range(world_size):
                ctx.Process(target=demo_fn, args=(rank, world_size, results_queue, *arguments),
                            kwargs=kwargs).start()

            res = []
            for rank in range(world_size):
                res.append(results_queue.get())

            done.set()

            return res

        if self.world_size > 1:
            logger.info(f'Initializing {self.world_size} parallel workers')

            if self.hparams.mp_port == 'random' or check_if_port_is_available(self.hparams.mp_port):
                self.hparams.mp_port = find_free_port()

            logger.info(f'Multiprocessing port is: {self.hparams.mp_port}')

            return _run(run_worker, self.world_size)
        else:
            logger.info(f'Single worker mode')
            return run_worker(0, 1, None, *arguments, **kwargs)

    def writer_cleanup(self):

        if self.writer is not None:
            self.writer.close()
            self.writer = None


