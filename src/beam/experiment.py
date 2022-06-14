import re
import sys
import time
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
from torch.utils.tensorboard import SummaryWriter
from shutil import copytree
import torch
import copy
import shutil
from collections import defaultdict
from .utils import include_patterns, logger
import pandas as pd
import torch.multiprocessing as mp
from .utils import setup, cleanup, set_seed, find_free_port, check_if_port_is_available, is_notebook
import torch.distributed as dist
from ray import tune
import optuna
from functools import partial
import ray

done = mp.Event()


def default_runner(rank, world_size, experiment, algorithm_generator, *args, **kwargs):
    alg = algorithm_generator(*args, **kwargs)
    save_model_results_arguments = kwargs[
        'save_model_results_arguments'] if 'save_model_results_arguments' in kwargs else None

    experiment.writer_control(enable=not (bool(rank)))
    for results in iter(alg):
        experiment.save_model_results(results, alg,
                                      print_results=experiment.print_results,
                                      visualize_results=experiment.visualize_results,
                                      store_results=experiment.store_results, store_networks=experiment.store_networks,
                                      visualize_weights=experiment.visualize_weights,
                                      argv=save_model_results_arguments)

    if world_size == 1:
        return alg, results

# check

def run_worker(rank, world_size, results_queue, job, experiment, *args):
    logger.info(f"Worker: {rank + 1}/{world_size} is running...")

    if world_size > 1:
        setup(rank, world_size, port=experiment.mp_port)

    experiment.set_rank(rank, world_size)
    set_seed(seed=experiment.seed, constant=rank, increment=False, deterministic=experiment.deterministic)

    res = job(rank, world_size, experiment, *args)

    if world_size > 1:

        cleanup(rank, world_size)
        results_queue.put({'rank': rank, 'results': res})
        done.wait()

    else:
        return res


class Study(object):

    def __init__(self, algorithm_generator, args):

        args.reload = False
        args.override = False
        args.print_results = False
        args.visualize_weights = False
        args.enable_tqdm = False

        start_time = time.time()
        exptime = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        args.identifier = f'{args.identifier}_hp_optimization_{exptime}'

        self.ag = algorithm_generator
        self.args = args

        logger.info('Hyperparameter Optimization')
        logger.info(f"beam project: {args.project_name}")
        logger.info('Experiment Hyperparameters')

        for k, v in vars(args).items():
            logger.info(k + ': ' + str(v))

    def runner_tune(self, config):

        args = copy.deepcopy(self.args)

        for k, v in config.items():
            setattr(args, k, v)

        experiment = Experiment(args, results_names='objective', hpo='tune', print_hyperparameters=False)
        alg, results = experiment(self.ag, return_results=True)

        if 'objective' in results:
            if type('objective') is tuple:
                return results['objective']
            elif issubclass(type(results['objective']), dict):
                tune.report(**results['objective'])
            else:
                return results['objective']

    def runner_optuna(self, trial, suggest):

        config = suggest(trial)

        logger.info('Next Hyperparameter suggestion:')
        for k, v in config.items():
            logger.info(k + ': ' + str(v))

        args = copy.deepcopy(self.args)

        for k, v in config.items():
            setattr(args, k, v)

        experiment = Experiment(args, hpo='optuna', results_names='objective',
                                trial=trial, print_hyperparameters=False)
        alg, results = experiment(self.ag, return_results=True)

        if 'objective' in results:
            if type('objective') is tuple:
                return results['objective']
            elif issubclass(type(results['objective']), dict):
                tune.report(**results['objective'])
            else:
                return results['objective']

    def tune(self, config, *args, **kwargs):

        analysis = tune.run(self.runner_tune, config=config, *args, **kwargs)

        return analysis

    def optuna(self, suggest, storage=None, sampler=None, pruner=None, study_name=None, direction=None,
               load_if_exists=False, directions=None, *args, **kwargs):

        if study_name is None:
            study_name = f'{self.args.project_name}/{self.args.algorithm}/{self.args.identifier}'

        runner = partial(self.runner_optuna, suggest=suggest)
        study = optuna.create_study(storage=storage, sampler=sampler, pruner=pruner, study_name=study_name,
                                    direction=direction, load_if_exists=load_if_exists, directions=directions)
        study.optimize(runner, *args, **kwargs)

        return study


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

        set_seed(args.seed)

        # torch.set_num_threads(100)

        if print_hyperparameters:
            logger.info(f"beam project: {args.project_name}")
            logger.info('Experiment Hyperparameters')

        self.args = vars(args)
        for k, v in vars(args).items():
            if print_hyperparameters:
                logger.info(k + ': ' + str(v))
            setattr(self, k, v)

        self.hpo = hpo
        self.trial = trial
        # determine the batch size

        torch.backends.cudnn.benchmark = self.cudnn_benchmark
        # parameters

        self.start_time = time.time()
        self.exptime = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self.device = torch.device(int(self.device) if self.device.isnumeric() else self.device)

        self.base_dir = os.path.join(self.root_dir, self.project_name, self.algorithm, self.identifier)
        os.makedirs(self.base_dir, exist_ok=True)

        self.exp_name = None
        self.load_model = False

        pattern = re.compile("\A\d{4}_\d{8}_\d{6}\Z")
        exp_names = os.listdir(self.base_dir)
        exp_indices = np.array([int(d.split('_')[0]) for d in exp_names if re.match(pattern, d) is not None])

        if self.reload:

            if self.resume >= 0:

                ind = np.nonzero(exp_indices == self.resume)[0]
                if len(ind):
                    self.exp_name = exp_names[ind]
                    self.exp_num = self.resume
                    self.load_model = True

            else:
                if len(exp_indices):
                    ind = np.argmax(exp_indices)
                    self.exp_name = exp_names[ind]
                    self.exp_num = exp_indices[ind]
                    self.load_model = True

        else:

            if self.override and len(exp_indices):

                ind = np.argmax(exp_indices)
                self.exp_name = exp_names[ind]
                self.exp_num = exp_indices[ind]
            else:
                self.override = False

        if self.exp_name is None:
            self.exp_num = np.argmax(exp_indices) + 1 if len(exp_indices) else 0
            self.exp_name = "%04d_%s" % (self.exp_num, self.exptime)

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

            if not self.override:
                logger.info("Creating new experiment")

            else:
                logger.info("Deleting old experiment")

                shutil.rmtree(self.root)
                self.exp_name = "%04d_%s" % (self.exp_num, self.exptime)
                self.root = os.path.join(self.base_dir, self.exp_name)

                # set dirs
                self.tensorboard_dir = os.path.join(self.root, 'tensorboard')
                self.checkpoints_dir = os.path.join(self.root, 'checkpoints')
                self.results_dir = os.path.join(self.root, 'results')
                self.code_dir = os.path.join(self.root, 'code')

            logger.info(f"Experiment directory is: {self.root}")

            os.makedirs(self.root)
            os.makedirs(self.tensorboard_dir)
            os.makedirs(self.checkpoints_dir)
            os.makedirs(self.results_dir)

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

            pd.to_pickle(vars(args), os.path.join(self.root, "args.pkl"))

        self.epoch = 0
        self.writer = None
        self.rank = 0
        self.world_size = self.parallel

        if self.world_size > 1:
            torch.multiprocessing.set_sharing_strategy('file_system')

        self.ddp = False

        # update experiment parameters

        if self.batch_size_train is None:
            self.batch_size_train = self.batch_size

        if self.batch_size_eval is None:
            self.batch_size_eval = self.batch_size

        if self.batch_size is None:
            self.batch_size = self.batch_size_train

        if self.epoch_length_train is None:
            self.epoch_length_train = self.epoch_length

        if self.epoch_length_eval is None:
            self.epoch_length_eval = self.epoch_length

    def reload_checkpoint(self, alg, iloc=-1):

        checkpoints = os.listdir(self.checkpoints_dir)
        checkpoints = pd.DataFrame({'name': checkpoints, 'index': [int(c.split('_')[-1]) for c in checkpoints]})
        checkpoints = checkpoints.sort_values('index')

        chp = checkpoints.iloc[iloc]['name']
        path = os.path.join(self.checkpoints_dir, chp)

        logger.info(f"Reload experiment from checkpoint: {path}")

        alg.load_checkpoint(path)

    def set_rank(self, rank, world_size):

        self.rank = rank
        self.world_size = world_size
        self.ddp = self.world_size > 1

        if self.device.type != 'cpu' and world_size > 1:
            self.device = rank

    def writer_control(self, enable=True, add_hyperparameters=True, networks=None, inputs=None):

        if enable and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.tensorboard_dir, comment=self.identifier)

        if not enable:
            self.writer = None

        if add_hyperparameters and self.writer is not None:
            self.writer.add_hparams(self.args, {}, run_name=self.exp_name)

        if networks is not None and self.writer is not None:
            for k, net in networks.items():
                self.writer.add_graph(net, inputs[k])

    def save_model_results(self, results, algorithm, visualize_results='yes',
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

        self.epoch += 1

        if not self.rank:

            if print_results:
                print()
                logger.info(f'Finished epoch {self.epoch}/{algorithm.n_epochs}:')

            decade = int(np.log10(self.epoch) + 1)
            logscale = not (self.epoch - 1) % (10 ** (decade - 1))

            for subset, res in results.items():

                if store_results == 'yes' or store_results == 'logscale' and logscale:
                    pd.to_pickle(res, os.path.join(self.results_dir, subset, f'results_{self.epoch:06d}'))

                alg = algorithm if visualize_weights else None

            if visualize_results == 'yes' or visualize_results == 'logscale' and logscale:
                self.log_data(copy.deepcopy(results), self.epoch, print_log=print_results, alg=alg, argv=argv)

            checkpoint_file = os.path.join(self.checkpoints_dir, f'checkpoint_{self.epoch:06d}')
            algorithm.save_checkpoint(checkpoint_file)

            if store_networks == 'no' or store_networks == 'logscale' and not logscale:
                os.remove(os.path.join(self.checkpoints_dir, f'checkpoint_{self.epoch - 1:06d}'))

        if self.world_size > 1:
            dist.barrier()

    def log_data(self, results, n, print_log=True, alg=None, argv=None):

        for subset, res in results.items():

            report = None
            if issubclass(type(res), dict):
                if 'scalar' in res:
                    report = res['scalar']
                else:
                    report = res

            if report is not None:
                for param, val in report.items():
                    if type(val) is dict or type(val) is defaultdict:
                        for p, v in val.items():
                            val[p] = np.mean(v)
                    elif isinstance(report[param], torch.Tensor):
                        report[param] = torch.mean(val)
                    else:
                        report[param] = np.mean(val)

                if print_log:
                    logger.info(f'{subset}:')
                    for param in report:
                        if not (type(report[param]) is dict or type(
                                report[param]) is defaultdict):
                            logger.info('%s %g \t|' % (param, report[param]))

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


    def __call__(self, algorithm_generator, *args, return_results=False, reload_results=False, **kwargs):

        res = self.run(default_runner, *(algorithm_generator, self, *args), **kwargs)

        if res is None or self.world_size > 1:
            alg = algorithm_generator(self, *args, **kwargs)
            self.reload_checkpoint(alg)

            if reload_results:
                results = {}
                for subset in os.listdir(alg.results_dir):

                    res = os.listdir(os.path.join(alg.results_dir, subset))
                    res = pd.DataFrame({'name': res, 'index': [int(c.split('_')[-1]) for c in res]})
                    res = res.sort_values('index')

                    res = res.iloc[iloc]['name']
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

        if self.parallel > 1:
            logger.info(f'Initializing {self.parallel} parallel workers')

            if self.mp_port is None or check_if_port_is_available(self.mp_port):
                self.mp_port = find_free_port()

            logger.info(f'Multiprocessing port is: {self.mp_port}')

            return _run(run_worker, self.parallel)
        else:
            logger.info(f'Single worker mode')
            return run_worker(0, 1, None, *arguments, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.tensorboard:
            self.writer.export_scalars_to_json(os.path.join(self.tensorboard_dir, "all_scalars.json"))
            self.writer.close()
