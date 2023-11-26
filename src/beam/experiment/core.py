import re
import sys
import time
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from shutil import copytree
import torch
import copy
import pandas as pd
import torch.multiprocessing as mp
import inspect
import torch.distributed as dist
from argparse import Namespace
from functools import partial
import atexit
import traceback


from ..utils import (setup_distributed, cleanup, set_seed, find_free_port, check_if_port_is_available, is_notebook,
                    find_port, as_numpy, lazy_property, include_patterns, check_type, beam_device, rmtree)
from ..path import beam_path, BeamPath, beam_key
from ..logger import beam_logger as logger
from ..config import get_beam_llm, print_beam_hyperparameters, BeamHparams


done = mp.Event()


def gen_hparams_string(experiment_path):
    experiment_path = beam_path(experiment_path)
    tensorboard_hparams = BeamHparams.from_path(experiment_path.joinpath('args.pkl'))
    tensorboard_hparams_keys = tensorboard_hparams.model_parameter + tensorboard_hparams.tune_parameter
    return '/'.join([f"{k}_{tensorboard_hparams[k]}" for k in tensorboard_hparams_keys])


def path_depth(path):

    if isinstance(path, str):
        path = beam_path(path)

    return len(str(path.resolve()).split(os.sep))


def beam_algorithm_generator(experiment, Alg, Dataset=None, alg_args=None, alg_kwargs=None,
                             dataset_args=None, dataset_kwargs=None):

    if alg_args is None:
        alg_args = tuple()
    if alg_kwargs is None:
        alg_kwargs = dict()
    if dataset_args is None:
        dataset_args = tuple()
    if dataset_kwargs is None:
        dataset_kwargs = dict()

    if Dataset is not None and not isinstance(Dataset, dict):
        datasets = {'dataset': Dataset}
    else:
        datasets = Dataset

    if datasets is not None:
        for k, v in datasets.items():
            if inspect.isclass(v):
                datasets[k] = v(experiment.hparams, *dataset_args, **dataset_kwargs)
            elif inspect.isfunction(v):
                datasets[k] = v(experiment.hparams, *dataset_args, **dataset_kwargs)

    if inspect.isclass(Alg):

        alg = Alg(experiment.hparams, *alg_args, **alg_kwargs)
        # if a new algorithm is generated, we clean the tensorboard writer. If the reload option is True,
        # the algorithm will fix the epoch number s.t. tensorboard graphs will not overlap
        experiment.writer_cleanup()
    else:

        alg = Alg

    alg.experiment = experiment

    if datasets is not None:
        alg.load_datasets(datasets)

    return alg


def default_runner(rank, world_size, experiment, algorithm_generator, *args, tensorboard_arguments=None, **kwargs):
    alg = algorithm_generator(*args, **kwargs)

    experiment.writer_control(enable=not (bool(rank)))
    results = {}

    t0 = time.time()

    try:
        for i, results in enumerate(iter(alg)):

            total_time = time.time() - t0
            estimated_time = total_time * (alg.hparams.n_epochs - i - 1) / (i + 1)

            experiment.save_model_results(copy.deepcopy(results), alg, i,
                                          print_results=experiment.hparams.print_results,
                                          visualize_results=experiment.hparams.visualize_results,
                                          store_results=experiment.hparams.store_results, store_networks=experiment.hparams.store_networks,
                                          visualize_weights=experiment.hparams.visualize_weights,
                                          argv=tensorboard_arguments, total_time=total_time, estimated_time=estimated_time)

    except KeyboardInterrupt as e:

        tb = traceback.format_exc()
        logger.warning(f"KeyboardInterrupt: Training was interrupted, Worker terminates.")
        logger.debug(f"KeyboardInterrupt: {e}")
        logger.debug(f"KeyboardInterrupt: {tb}")

        if rank == 0:
            checkpoint_file = experiment.checkpoints_dir.joinpath(f'checkpoint_{alg.epoch + 1:06d}')
            alg.save_checkpoint(checkpoint_file)

    except Exception as e:

        tb = traceback.format_exc()

        llm = get_beam_llm() if experiment.llm is None else experiment.llm

        if llm is not None:
            explain = llm.explain_traceback(tb)
            logger.error(f"LLM Message: {explain}")

        if not is_notebook():
            raise e

        logger.error(f"Exception: Training was interrupted, Worker terminates, but checkpoint will be saved.")
        logger.error(f"Exception: {e}")
        logger.error(f"Exception: {tb}")

        if rank == 0:
            checkpoint_file = experiment.checkpoints_dir.joinpath(f'checkpoint_{alg.epoch + 1:06d}')
            alg.save_checkpoint(checkpoint_file)

    if world_size == 1:
        return alg, results


def run_worker(rank, world_size, results_queue, job, experiment, *args, **kwargs):

    logger.info(f"Worker: {rank + 1}/{world_size} is running...")

    if world_size > 1:
        backend = experiment.hparams.mp_backend
        if backend is None:
            backend = 'nccl' if experiment.hparams.device.type == 'cuda' else 'gloo'

        setup_distributed(rank, world_size, port=experiment.hparams.mp_port, backend=backend)

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

    def __init__(self, args, hpo=None, trial=None, print_hyperparameters=None, reload_iloc=-1,
                 reload_loc=None, reload_name=None):
        """

        @param args:
        @param hpo:
        @param trial:
        @param print_hyperparameters: If None, default behavior is to print hyperparameters only outside of jupyter notebooks
        """

        if print_hyperparameters is None:
            print_hyperparameters = not is_notebook()
        self.print_hyperparameters = print_hyperparameters

        self.tensorboard_hparams = {}

        if not isinstance(args, BeamHparams):
            args = BeamHparams(hparams=args)

        self.vars_args = dict(args.items())
        for k, v in self.vars_args.items():
            param_type = check_type(v)
            if param_type.major == 'scalar' and param_type.element in ['bool', 'str', 'int', 'float']:
                self.tensorboard_hparams[k] = v

        self.hparams = copy.deepcopy(args)
        set_seed(seed=self.hparams.seed, constant=0, increment=False, deterministic=self.hparams.deterministic)

        # parameters
        self.start_time = time.time()
        self.exptime = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self.hparams.device = beam_device(self.hparams.device)

        root_path = beam_path(self.hparams.root_dir)
        self.base_dir = root_path.joinpath(self.hparams.project_name, self.hparams.algorithm, self.hparams.identifier)
        self.base_dir.mkdir(exist_ok=True, parents=True)

        self.exp_name = None
        self.load_model = False

        pattern = re.compile("\A\d{4}_\d{8}_\d{6}\Z")
        exp_names = list(filter(lambda x: re.match(pattern, str(x)) is not None, self.base_dir.iterdir()))
        exp_indices = np.array([int(d.split('_')[0]) for d in exp_names])

        exp_num = None
        if self.hparams.reload:

            if type(self.hparams.resume) is str:

                if self.base_dir.joinpath(self.hparams.resume).is_dir():
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

        if self.hparams.reload and not self.load_model:
            logger.warning(f"Did not find existing experiment to match your specifications: basedir={self.base_dir} resume={self.hparams.resume}")

        if self.exp_name is None:
            exp_num = np.max(exp_indices) + 1 if len(exp_indices) else 0
            self.exp_name = "%04d_%s" % (exp_num, self.exptime)

        self.exp_num = exp_num
        # init experiment parameters
        self.root = self.base_dir.joinpath(self.exp_name)

        # set dirs
        self.tensorboard_dir = self.root.joinpath('tensorboard')
        self.checkpoints_dir = self.root.joinpath('checkpoints')
        self.results_dir = self.root.joinpath('results')
        self.code_dir = self.root.joinpath('code')

        if self.load_model:
            logger.cleanup()
            logger.add_file_handlers(self.root.joinpath('experiment.log'))
            logger.info(f"Resuming existing experiment")

        self.writer = None

        self.rank = 0
        self.world_size = args.parallel

        if self.world_size > 1:
            torch.multiprocessing.set_sharing_strategy('file_system')

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

        # build the hyperparamter class which will be sent to the dataset and algorithm classes

        if self.load_model:
            self.hparams.reload_path = self.reload_checkpoint(iloc=reload_iloc, loc=reload_loc, name=reload_name)
        else:
            self.hparams.reload_path = None

        self.trial = trial

        self.hparams.hpo = hpo
        self.hparams.ddp = False
        self.hparams.rank = self.rank
        self.hparams.world_size = self.world_size

        if self.hparams.device.type == 'cuda':
            if self.hparams.device_list is not None:
                self.hparams.device_list = [beam_device(di) for di in self.hparams.device_list]
            else:
                self.hparams.device_list = [beam_device(di+self.hparams.device.index) for di in range(self.hparams.parallel)]

        self.comet_exp = None
        self.comet_writer = None
        self.mlflow_writer = None
        self.root_dir_is_built = False

        atexit.register(self.cleanup)

    @lazy_property
    def llm(self):
        if self.hparams.llm is not None:
            from ..llm import beam_llm
            return beam_llm(self.hparams.llm)
        return None

    def __del__(self):
        self.cleanup()

    def cleanup(self):
        if self.comet_writer is not None:
            self.comet_writer.close()
            self.comet_writer = None

        if self.mlflow_writer is not None:
            self.mlflow_writer.close()
            self.mlflow_writer = None

        if self.writer is not None:
            self.writer.close()
            self.writer = None

    @staticmethod
    def reload_from_path(path, override_hparams=None, reload_iloc=-1, reload_loc=None, reload_name=None, **argv):

        path = beam_path(path)
        logger.info(f"Reload experiment from path: {path}")

        args = BeamHparams.from_path(path.joinpath('args.pkl'))
        args.override = False
        args.reload = True

        path, d = path.parent, path.name
        if not d:
            path, d = path.parent, path.name
        args.resume = d

        if override_hparams is not None:
            for k, v in override_hparams.items():
                if k in ['reload', 'resume', 'override', 'project', 'algorithm', 'identifier', 'root_dir']:
                    continue
                setattr(args, k, v)

        return Experiment(args, reload_iloc=reload_iloc, reload_loc=reload_loc, reload_name=reload_name, **argv)

    def reload_checkpoint(self, alg=None, iloc=None, loc=None, name=None):

        if iloc is None and loc is None and name is None:
            if self.checkpoints_dir.joinpath('checkpoint_best').is_file():
                name = 'checkpoint_best'
            else:
                iloc = -1

        if name is not None:
            path = self.checkpoints_dir.joinpath(name)

        else:

            checkpoints = list(self.checkpoints_dir.iterdir())
            checkpoints = [c for c in checkpoints if str(c).split('_')[-1].isnumeric()]
            checkpoints_int = [int(str(c).split('_')[-1]) for c in checkpoints]

            if not(len(checkpoints)):
                logger.error(f"Directory of checkpoints does not contain valid checkpoint files")
                return

            checkpoints = pd.DataFrame({'name': checkpoints}, index=checkpoints_int)
            checkpoints = checkpoints.sort_index()

            if loc is not None:
                chp = str(checkpoints.loc[loc]['name'])
                path = self.checkpoints_dir.joinpath(chp)
            else:
                chp = str(checkpoints.iloc[iloc]['name'])
                path = self.checkpoints_dir.joinpath(chp)

        logger.info(f"Reload experiment from checkpoint: {path}")

        if alg is not None:
            alg.load_checkpoint(path, hparams=False)

        else:
            return path

    def set_rank(self, rank, world_size):

        self.rank = rank
        self.world_size = world_size

        self.hparams.rank = rank
        self.hparams.world_size = world_size

        self.hparams.ddp = self.world_size > 1
        self.hparams.enable_tqdm = self.hparams.enable_tqdm and (rank == 0)

        if self.hparams.device.type != 'cpu' and world_size > 1:
            self.hparams.device = beam_device(self.hparams.device_list[rank])

        logger.info(f'Worker {rank + 1} will be running on device={str(self.hparams.device)}')

    def writer_control(self, enable=True, networks=None, inputs=None):

        if enable and self.writer is None and self.hparams.tensorboard:
            if isinstance(self.tensorboard_dir, BeamPath):
                from tensorboardX import SummaryWriter
                self.writer = SummaryWriter(log_dir=str(self.tensorboard_dir.joinpath('logs')),
                                            comment=self.hparams.identifier)
            else:
                logger.warning(f"Tensorboard directory is not a BeamPath object. Tensorboard will not be enabled.")

        if self.hparams.comet:

            import comet_ml
            # from comet_ml.integration.pytorch import log_model

            api_key = self.hparams.COMET_API_KEY
            if api_key is None:
                # api_key = os.environ.get('COMET_API_KEY', None)
                api_key = beam_key('COMET_API_KEY')
            git_directory = self.hparams.git_directory
            if git_directory is None and isinstance(self.code_dir, BeamPath):
                git_directory = str(self.code_dir)
            if git_directory is not None:
                os.environ['COMET_GIT_DIRECTORY'] = git_directory
                log_code = True
            else:
                log_code = False

            logger.info("Logging this experiment to comet.ml")

            from tensorboardX import SummaryWriter
            self.comet_writer = SummaryWriter(comet_config={'api_key': api_key,
                                                            'project_name': self.hparams.project_name,
                                                            'log_code': log_code,
                                                            'workspace': self.hparams.comet_workspace,
                                                            'disabled': False})

            self.comet_writer.add_hparams(self.tensorboard_hparams, {})

            self.comet_exp = self.comet_writer._get_comet_logger()._experiment
            self.comet_exp.add_tag(self.hparams.identifier)
            self.comet_exp.set_name(self.exp_name)

        if self.hparams.mlflow:

            from .beam_mlflow import MLflowSummaryWriter
            self.mlflow_writer = MLflowSummaryWriter(self.exp_name, self.tensorboard_hparams, self.hparams.mlflow_url)

        if networks is not None and enable:
            if self.writer is not None:
                for k, net in networks.items():
                    self.writer.add_graph(net, inputs[k])
            if self.comet_exp is not None:
                self.comet_exp.set_model_graph(str(networks))

    def upload(self, path, name=None, overwrite=False):
        pass
        # if name is None:
        #     name = path.name
        #
        # if self.hparams.upload == 'comet':
        #     self.comet_exp.log_asset(path, overwrite=overwrite)
        # elif self.hparams.upload == 'mlflow':
        #     self.mlflow_writer.log_artifact(path, name)
        # else:
        #     raise ValueError(f"Unknown upload method: {self.hparams.upload}")

    def save_model_results(self, reporter, algorithm, iteration, visualize_results='yes',
                           store_results='logscale', store_networks='logscale', print_results=True,
                           visualize_weights=False, argv=None, total_time=None, estimated_time=None):

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
                reporter.print_metadata()

            decade = int(np.log10(epoch) + 1)
            logscale = not (epoch - 1) % (self.hparams.visualize_results_log_base ** (decade - 1))

            if store_results == 'yes' or store_results == 'logscale' and logscale:

                self.results_dir.mkdir(parents=True, exist_ok=True)
                reporter.write_to_path(self.results_dir.joinpath(f'results_{epoch:06d}'))

            alg = algorithm if visualize_weights else None

            if iteration+1 == algorithm.n_epochs or visualize_results == 'yes' or visualize_results == 'logscale' and logscale:
                self.log_data(reporter, epoch, print_log=print_results, alg=alg, argv=argv)

            checkpoint_file = self.checkpoints_dir.joinpath(f'checkpoint_{epoch:06d}')
            algorithm.save_checkpoint(checkpoint_file)

            if algorithm.best_state:
                checkpoint_file = self.checkpoints_dir.joinpath(f'checkpoint_best')
                algorithm.save_checkpoint(checkpoint_file)

            if store_networks == 'no' or store_networks == 'logscale' and not logscale:
                try:
                    self.checkpoints_dir.joinpath(f'checkpoint_{epoch - 1:06d}').unlink()
                except OSError:
                    pass

        if self.world_size > 1:
            dist.barrier()

    def log_data(self, reporter, n, print_log=True, alg=None, argv=None):

        if print_log:
            reporter.print_stats()

        if self.writer is not None:
            logger.info(f"Tensorboard results are stored to: {self.root}")
            reporter.write_to_tensorboard(self.writer, hparams=self.tensorboard_hparams)
        if self.comet_writer is not None:
            logger.info(f"Comet results are stored to: {self.comet_exp.get_key()}")
            reporter.write_to_tensorboard(self.comet_writer, hparams=self.tensorboard_hparams)
        if self.mlflow_writer is not None:
            logger.info(f"MLFlow results are stored to: {self.mlflow_writer.url}")
            reporter.write_to_tensorboard(self.mlflow_writer, hparams=self.tensorboard_hparams)

        def write_network(writer, net, name, n):

            if writer is not None:
                writer.add_histogram("weight_%s/%s" % (net, name), as_numpy(param), n,
                                          bins='tensorflow')
                writer.add_histogram("grad_%s/%s" % (net, name), as_numpy(param.grad), n,
                                          bins='tensorflow')
                if hasattr(param, 'intermediate'):
                    writer.add_histogram("iterm_%s/%s" % (net, name), as_numpy(param.intermediate),
                                              n,
                                              bins='tensorflow')

        if alg is not None:
            networks = alg.networks
            for net in networks:
                for name, param in networks[net].named_parameters():
                    try:
                        write_network(self.writer, net, name, n)
                    except:
                        pass
                    try:
                        write_network(self.comet_writer, net, name, n)
                    except:
                        pass

    @staticmethod
    def _tensorboard(port=None, get_port_from_beam_port_range=True, base_dir=None, log_dirs=None, hparams=False):

        port = find_port(port=port, get_port_from_beam_port_range=get_port_from_beam_port_range)
        if port is None:
            return

        logger.info(f"Opening a tensorboard server on port: {port}")

        if hparams:
            command_argument = f"--bind_all --logdir {base_dir} --port {port}"
        else:
            command_argument = f"--bind_all --logdir_spec={log_dirs} --port {port}"
        from tensorboard.notebook import start as start_tensorboard
        start_tensorboard(command_argument)

    def normalize_experiment_path(self, path, level=0):

        normal_path = [self.hparams.root_dir, self.hparams.project_name,
                       self.hparams.algorithm, self.hparams.identifier]
        pd = path_depth(self.hparams.root_dir)

        return os.path.join(*normal_path[:len(normal_path)-pd-level], path)

    @staticmethod
    def open_tensorboard(root='', project=None, algorithm=None,
                         identifier=None, experiment=None, hparams=False, port=None,
                         get_port_from_beam_port_range=True):
        depth = 4
        filters = {'project': None, 'algorithm': None, 'identifier': None, 'experiment':None}
        if project is not None:
            path_type = check_type(project)
            if path_type.minor == 'list':
                filters['project'] = project
            else:
                filters['project'] = [project]
                path = os.path.join(root, project)
                if os.path.isdir(path):
                    root = path
                    depth = 3

        if algorithm is not None:
            path_type = check_type(algorithm)
            if path_type.minor == 'list':
                filters['algorithm'] = algorithm
            else:
                filters['algorithm'] = [algorithm]
                path = os.path.join(root, algorithm)
                if os.path.isdir(path):
                    root = path
                    depth = 2

        if identifier is not None:
            path_type = check_type(identifier)
            if path_type.minor == 'list':
                filters['identifier'] = identifier
            else:
                filters['identifier'] = [identifier]
                path = os.path.join(root, identifier)
                if os.path.isdir(path):
                    root = path
                    depth = 1
        if experiment is not None:
            path_type = check_type(experiment)
            if path_type.minor == 'list':
                filters['experiment'] = experiment
            else:
                filters['experiment'] = [experiment]
                path = os.path.join(root, experiment)
                if os.path.isdir(path):
                    root = path
                    depth = 0

        experiments = [d[0] for d in list(os.walk(root)) if (path_depth(d[0]) - path_depth(root)) == depth]
        experiments = [os.path.normpath(e) for e in experiments]

        if filters['project'] is not None:
            experiments = list(filter(lambda x: x.split(os.sep)[-4] in filters['project'], experiments))
        if filters['algorithm'] is not None:
            experiments = list(filter(lambda x: x.split(os.sep)[-3] in filters['algorithm'], experiments))
        if filters['identifier'] is not None:
            experiments = list(filter(lambda x: x.split(os.sep)[-2] in filters['identifier'], experiments))
        if filters['experiment'] is not None:
            experiments = list(filter(lambda x: x.split(os.sep)[-1] in filters['experiment'], experiments))

        names = ['/'.join(e.split(os.sep)[-3:]) for e in experiments]
        names = [f"{n}/{gen_hparams_string(e)}" for n, e in zip(names, experiments)]

        experiments = [os.path.join(e, 'tensorboard', 'logs') for e in experiments]
        log_dirs = ','.join([f"{n}:{e}" for n, e in zip(names, experiments)])

        Experiment._tensorboard(port=port, get_port_from_beam_port_range=get_port_from_beam_port_range,
                                base_dir=root, log_dirs=log_dirs, hparams=hparams)

    def tensorboard(self, port=None, add_all_of_same_identifier=False, add_all_of_same_algorithm=False,
                          add_all_of_same_project=False, more_experiments=None, more_identifiers=None,
                          more_algorithms=None, get_port_from_beam_port_range=True, hparams=False):

        suffix = 'hparams' if hparams else 'logs'

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

        #TODO: add support for beampath
        base_dir = str(base_dir)
        experiments = [d[0] for d in list(os.walk(base_dir)) if (path_depth(d[0]) - path_depth(base_dir)) == depth]

        if more_experiments is not None:
            if hparams:
                logger.error("hparams visualization does not support adding additional experiments")
            if type(more_experiments) is str:
                more_experiments = [more_experiments]
                experiments = experiments + [self.normalize_experiment_path(e, level=0) for e in more_experiments]

        if more_identifiers is not None:
            if hparams:
                logger.error("hparams visualization does not support adding additional experiments")
            if type(more_identifiers) is str:
                more_identifiers = [more_identifiers]
                depth = 1
                for identifier in more_identifiers:
                    identifier = self.normalize_experiment_path(identifier, level=depth)
                    experiments = experiments + [d[0] for d in list(os.walk(identifier)) if (path_depth(d[0]) - path_depth(identifier)) == depth]

        if more_algorithms is not None:
            if hparams:
                logger.error("hparams visualization does not support adding additional experiments")
            if type(more_algorithms) is str:
                more_algorithms = [more_algorithms]
                depth = 2
                for algorithm in more_algorithms:
                    algorithm = self.normalize_experiment_path(algorithm, level=depth)
                    experiments = experiments + [d[0] for d in list(os.walk(algorithm)) if (path_depth(d[0]) - path_depth(algorithm)) == depth]

        experiments = [os.path.normpath(e) for e in experiments]
        names = ['/'.join(e.split(os.sep)[-3:]) for e in experiments]
        names = [f"{n}/{gen_hparams_string(e)}" for n, e in zip(names, experiments)]

        experiments = [os.path.join(e, 'tensorboard', suffix) for e in experiments]
        log_dirs = ','.join([f"{n}:{e}" for n, e in zip(names, experiments)])

        self._tensorboard(port=port, get_port_from_beam_port_range=get_port_from_beam_port_range,
                          base_dir=base_dir, log_dirs=log_dirs, hparams=hparams)

    def algorithm_generator(self, Alg, Dataset=None, alg_args=None, alg_kwargs=None,
                             dataset_args=None, dataset_kwargs=None):
        return beam_algorithm_generator(self, Alg=Alg, Dataset=Dataset, alg_args=alg_args, alg_kwargs=alg_kwargs,
                             dataset_args=dataset_args, dataset_kwargs=dataset_kwargs)

    def fit(self, Alg=None, Dataset=None, *args, algorithm_generator=None, return_results=False, reload_results=False,
            tensorboard_arguments=None, alg_args=None, alg_kwargs=None, dataset_args=None,
            dataset_kwargs=None, **kwargs):

        if algorithm_generator is None:
            ag = partial(beam_algorithm_generator, Alg=Alg, Dataset=Dataset, alg_args=alg_args, alg_kwargs=alg_kwargs,
                                 dataset_args=dataset_args, dataset_kwargs=dataset_kwargs)
        else:

            if Alg is not None:
                ag = partial(algorithm_generator, Alg=Alg, Dataset=Dataset, alg_args=alg_args, alg_kwargs=alg_kwargs,
                                     dataset_args=dataset_args, dataset_kwargs=dataset_kwargs)
            else:
                ag = algorithm_generator

        return self(ag, *args, return_results=return_results, reload_results=reload_results,
                    tensorboard_arguments=tensorboard_arguments, **kwargs)

    def build_experiment_dir(self):

        logger.cleanup()
        logger.add_file_handlers(self.root.joinpath('experiment.log'))
        print_beam_hyperparameters(self.hparams, debug_only=not self.print_hyperparameters)

        self.hparams.to_path(self.root.joinpath('hparams.pkl'))

        if not self.hparams.override:
            logger.info(f"Creating new experiment")

        else:
            logger.warning("Deleting old experiment")

            rmtree(self.root)
            self.exp_name = "%04d_%s" % (self.exp_num, self.exptime)
            self.root = self.base_dir.joinpath(self.exp_name)

            # set dirs
            self.tensorboard_dir = self.root.joinpath('tensorboard')
            self.checkpoints_dir = self.root.joinpath('checkpoints')
            self.results_dir = self.root.joinpath('results')
            self.code_dir = self.root.joinpath('code')

        logger.info(f"Experiment directory is: {self.root}")

        self.tensorboard_dir.joinpath('logs').mkdir(exist_ok=True, parents=True)
        self.tensorboard_dir.joinpath('hparams').mkdir(exist_ok=True, parents=True)
        self.checkpoints_dir.mkdir(exist_ok=True, parents=True)

        # make log dirs
        self.results_dir.mkdir(exist_ok=True, parents=True)

        # copy code to dir
        if is_notebook():
            code_root_path = os.getcwd()
        else:
            code_root_path = sys.argv[0]

        self.source_dir = os.path.dirname(os.path.realpath(code_root_path))
        #TODO: handle the case where root_path is not BeamPath object
        if self.hparams.copy_code and isinstance(self.code_dir, BeamPath):
            copytree(self.source_dir, str(self.code_dir),
                     ignore=include_patterns('*.py', '*.md', '*.ipynb'))
        else:
            if not isinstance(self.code_dir, BeamPath):
                logger.warning("Code directory is not BeamPath object. Skipping code copy.")

        self.root.joinpath('args.pkl').write(self.vars_args)
        self.root_dir_is_built = True

    def __call__(self, algorithm_generator, *args, return_results=False, reload_results=False,
                  tensorboard_arguments=None, **kwargs):

        if not self.load_model and not self.root_dir_is_built:
            self.build_experiment_dir()

        try:
            res = self.run(default_runner, *(algorithm_generator, self, *args),
                           tensorboard_arguments=tensorboard_arguments, **kwargs)

        except KeyboardInterrupt:

            res = None
            logger.warning(f"KeyboardInterrupt: Training was interrupted, reloads last checkpoint")

        if res is None or self.world_size > 1:
            alg = algorithm_generator(self, *args, **kwargs)
            self.reload_checkpoint(alg)

            if reload_results:
                results = {}
                for subset in alg.results_dir.iterdir():

                    res = list(subset.iterdir())
                    res = pd.DataFrame({'name': res, 'index': [int(str(c).split('_')[-1]) for c in res]})
                    res = res.sort_values('index')

                    res = res.iloc['name']
                    path = alg.results_dir.joinpath(subset, res)
                    results[subset] = path

                if reload_results:
                    results = {subset: path.read() for subset, path in results.items()}

        else:
            alg, results = res

        if return_results:
            return alg, results
        else:
            return alg

    def run(self, job, *args, **kwargs):

        arguments = (job, self, *args)

        def _run(demo_fn, world_size):

            ctx = mp.get_context(self.hparams.mp_context)
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
            logger.warning(f"Caution: Sometimes DDP experiments can fail due to a bad configuration. "
                           f"Specifically, if in_place error set --no-broadcast-buffer flag and for subgraph issues"
                           f"set --find-unused-parameters")

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

        if self.comet_writer is not None:
            self.comet_writer.close()
            self.comet_writer = None



