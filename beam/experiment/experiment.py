import re
import sys
import time
import numpy as np
import os
import warnings
import torch
import copy
import pandas as pd
import torch.multiprocessing as mp

from ..base import base_paths
from ..type import Types
from ..utils import (cached_property, set_seed, find_free_port, check_if_port_is_available, is_notebook,
                     find_port, as_numpy, check_type, beam_device, beam_service_port)
from ..path import beam_path, BeamPath, beam_key
from ..logging import beam_logger as logger
from ..config import print_beam_hyperparameters, BeamConfig, UniversalConfig
from .utils import (path_depth, gen_hparams_string, nn_algorithm_generator, default_runner, simple_runner, run_worker,
                    build_device_list, simple_algorithm_generator)

warnings.filterwarnings('ignore', category=FutureWarning)


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

    def __init__(self, args, hpo=None, trial=None, print_hyperparameters=None, reload_iloc=None,
                 reload_dir=None, reload_loc=None, reload_name=None):
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

        if not isinstance(args, BeamConfig):
            args = BeamConfig(config=args)

        self.vars_args = dict(args.items())
        for k, v in self.vars_args.items():
            param_type = check_type(v)
            if param_type.major == Types.scalar and param_type.element in [Types.bool, Types.str, Types.int, Types.float]:
                self.tensorboard_hparams[k] = v

        self.hparams = copy.deepcopy(args)

        set_seed(seed=self.hparams.seed, constant=0, increment=False, deterministic=self.hparams.deterministic)

        # parameters
        self.start_time = time.time()
        self.exptime = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self.device = beam_device(self.hparams.device)
        self.device_list = None

        self.exp_name = None
        self.load_model = False
        self.log_experiment = self.hparams.get('log_experiment', default=True)

        if not self.log_experiment:
            self.experiment_dir = beam_path(base_paths.projects_experiments)
            self.exp_name = 'experiment'
            self.exp_num = None
            self.load_model = False

        elif reload_dir is None:
            root_path = beam_path(self.hparams.logs_path)
            base_dir = root_path.joinpath(self.hparams.project_name, self.hparams.algorithm, self.hparams.identifier)

            pattern = re.compile(r"\A\d{6}_\d{8}_\d{6}\Z")

            if base_dir.exists():
                assert base_dir.is_dir(), f"Experiment directory contains an existing file: {base_dir}"
                exp_names = list(filter(lambda x: re.match(pattern, x.name) is not None, base_dir.iterdir()))
                exp_indices = np.array([int(d.name.split('_')[0]) for d in exp_names])
            else:
                exp_names = []
                exp_indices = np.array([])

            exp_num = None
            if self.hparams.reload:

                if type(self.hparams.resume) is str:

                    if base_dir.joinpath(self.hparams.resume).is_dir():
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
                    self.hparams.set('override', False)

            if self.hparams.reload and not self.load_model:
                logger.warning(f"Did not find existing experiment to match your specifications: basedir={base_dir} resume={self.hparams.resume}")

            if self.exp_name is None:
                exp_num = np.max(exp_indices) + 1 if len(exp_indices) else 0
                self.exp_name = "%06d_%s" % (exp_num, self.exptime)

            self.exp_num = exp_num
            # init experiment parameters
            self.experiment_dir = base_dir.joinpath(self.exp_name)

        else:
            self.experiment_dir = beam_path(reload_dir)
            self.exp_name = self.experiment_dir.name
            self.exp_num = None
            self.load_model = True

        # set dirs
        self.tensorboard_dir = self.experiment_dir.joinpath('tensorboard')
        self.checkpoints_dir = self.experiment_dir.joinpath('checkpoints')
        self.results_dir = self.experiment_dir.joinpath('results')
        self.code_dir = self.experiment_dir.joinpath('code')

        self.store_init_path = None
        if self.hparams.get('store_init_args', default=True):
            self.store_init_path = self.experiment_dir.joinpath('init_alg_args.yaml')

        if self.load_model:
            logger.cleanup(clean_default=False)
            logger.add_file_handlers(self.experiment_dir.joinpath('experiment.log'))
            logger.info(f"Resuming existing experiment")

        self.rank = 0
        self.world_size = args.n_gpus
        if hasattr(args, 'n_gpus_per_worker') and args.n_gpus_per_worker is not None:
            self.world_size = self.world_size // args.n_gpus_per_worker
            assert self.world_size * args.n_gpus_per_worker == args.n_gpus, \
                (f"Total number of gpus ({args.n_gpus}) is not divisible by number of gpus per worker "
                 f"({args.n_gpus_per_worker})")

        if self.world_size > 1:
            torch.multiprocessing.set_sharing_strategy('file_system')

        # fill the batch size

        if 'batch_size' in self.hparams:
            if self.hparams.batch_size_train is None:
                self.hparams.set('batch_size_train', self.hparams.batch_size)

            if self.hparams.batch_size_eval is None:
                self.hparams.set('batch_size_eval', self.hparams.batch_size)

            if self.hparams.batch_size is None:
                self.hparams.set('batch_size', self.hparams.batch_size_train)

        # build the hyperparamter class which will be sent to the dataset and algorithm classes

        if self.load_model:

            if reload_iloc is None and reload_loc is None and reload_name is not None:

                if self.hparams.reload_checkpoint == 'last':
                    reload_iloc, reload_loc, reload_name = -1, None, None
                elif self.hparams.reload_checkpoint != 'best':
                    try:
                        reload_iloc, reload_loc, reload_name = None, int(self.hparams.reload_checkpoint), None
                    except ValueError:
                        reload_iloc, reload_loc, reload_name = None, None, self.hparams.reload_checkpoint

            reload_path = self.reload_checkpoint(iloc=reload_iloc, loc=reload_loc, name=reload_name)
        else:
            reload_path = None

        self.hparams.set('reload_path', reload_path)

        self.trial = trial

        self.hpo = hpo
        self.distributed_training = False

        if self.device.type == 'cuda':
            self.device_list = build_device_list(self.hparams)

        self.comet_exp = None
        self.tensorboard_writer = None
        self.comet_writer = None
        self.mlflow_writer = None
        self.logs_path_is_built = False
        self.source_dir = None

    def algorithm_generator(self, *args, **kwargs):
        return nn_algorithm_generator(self, *args, **kwargs)

    @cached_property
    def training_framework(self):
        return self.hparams.get('training_framework', 'torch')

    @cached_property
    def llm(self):
        if self.hparams.llm is not None:
            from ..llm import beam_llm
            return beam_llm(self.hparams.llm)
        return None

    @staticmethod
    def reload_from_path(path, override_hparams=None, reload_iloc=None, reload_loc=None, reload_name=None, **argv):

        path = beam_path(path)
        logger.info(f"Reload experiment from path: {path}")

        args = BeamConfig.from_path(path.joinpath('args.pkl'))
        args.override = False
        args.reload = True

        if override_hparams is not None:
            for k, v in override_hparams.items():
                if k in ['reload', 'resume', 'override', 'project', 'algorithm', 'identifier', 'logs_path']:
                    continue
                setattr(args, k, v)

        return Experiment(args, reload_iloc=reload_iloc, reload_loc=reload_loc, reload_name=reload_name,
                          reload_dir=path, **argv)

    def reload_best_checkpoint(self, alg=None):
            return self.reload_checkpoint(alg, name='checkpoint_best')

    def reload_checkpoint(self, alg=None, iloc=None, loc=None, name=None):

        if iloc is None and loc is None and name is None:
            if self.checkpoints_dir.joinpath('checkpoint_best').exists():
                name = 'checkpoint_best'
            else:
                iloc = -1

        if name is not None:
            path = self.checkpoints_dir.joinpath(name)

        else:

            checkpoints = list(self.checkpoints_dir.iterdir())
            checkpoints = [c for c in checkpoints if str(c).split('_')[-1].isnumeric()]
            checkpoints_int = [int(c.name.split('_')[-1]) for c in checkpoints]

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

        logger.info(f"Reload experiment from checkpoint: {path.name}")

        if alg is not None:
            alg.load_checkpoint(path, hparams=False)
            return alg
        else:
            return path

    def get_alg_init_args(self):

        try:
            init_args = self.store_init_path.read(ext='.pkl')
        except FileNotFoundError:
            return [], {}

        return init_args['args'], init_args['kwargs']

    def set_rank(self, rank, world_size, devices=None):

        self.rank = rank
        self.world_size = world_size

        self.distributed_training = self.world_size > 1
        self.hparams.set('enable_tqdm', self.hparams.enable_tqdm and (rank == 0))

        if self.device.type != 'cpu' and world_size > 1:
            if devices is None:
                self.device = beam_device(self.device_list[rank])
            else:
                self.device = beam_device(devices[0])
                self.device_list = devices

        logger.info(f'Worker {rank + 1} will be running on device={str(self.device)}')

    def writer_control(self, networks=None, inputs=None):

        if self.tensorboard_writer is None and (self.hparams.tensorboard and self.log_experiment):
            if isinstance(self.tensorboard_dir, BeamPath):
                from tensorboardX import SummaryWriter
                self.tensorboard_writer = SummaryWriter(log_dir=str(self.tensorboard_dir.joinpath('logs')),
                                                        comment=self.hparams.identifier)
            else:
                logger.warning(f"Tensorboard directory is not a BeamPath object. Tensorboard will not be enabled.")

        if self.hparams.comet:

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
            mlflow_url = self.hparams.mlflow_url
            if mlflow_url is None:
                mlflow_url = f"http://localhost:{beam_service_port('MLFLOW_PORT')}"
            mlflow_exp_name = '-'.join(self.experiment_dir.parts[-4:])
            self.mlflow_writer = MLflowSummaryWriter(mlflow_exp_name, self.tensorboard_hparams, mlflow_url)

        if networks is not None:
            if self.tensorboard_writer is not None:
                for k, net in networks.items():
                    self.tensorboard_writer.add_graph(net, inputs[k])
            if self.comet_exp is not None:
                self.comet_exp.set_model_graph(str(networks))

    def save_state(self, algorithm, name='best_checkpoint.bmpr', **kwargs):
        path = self.checkpoints_dir.joinpath(name)
        algorithm.save_state(path, **kwargs)

    def save_model_results(self, reporter, algorithm, iteration, visualize_results=None,
                           store_results=None, store_networks=None, print_results=None,
                           visualize_weights=None, argv=None, save_checkpoint=True):

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

        visualize_weights = visualize_weights or self.hparams.get('visualize_weights', default=False)
        visualize_results = visualize_results or self.hparams.get('visualize_results', default='yes')
        store_results = store_results or self.hparams.get('store_results', default='logscale')
        store_networks = store_networks or self.hparams.get('store_networks', default='logscale')
        print_results = print_results or self.hparams.get('print_results', default=True)

        epoch = algorithm.epoch
        if not self.rank:

            if print_results:
                reporter.print_metadata()

            decade = int(np.log10(epoch) + 1)
            logscale = not (epoch - 1) % (self.hparams.visualize_results_log_base ** (decade - 1))

            if ((iteration+1 == algorithm.n_epochs and visualize_results != 'never') or
                    store_results == 'yes' or (store_results == 'logscale' and logscale) or
                (store_results == 'best' and algorithm.best_state)):
                reporter.write_to_path(self.results_dir.joinpath(f'results_{epoch:06d}'))

            alg = algorithm if visualize_weights else None

            if ((iteration+1 == algorithm.n_epochs and visualize_results != 'never') or visualize_results == 'yes' or
                    (visualize_results == 'logscale' and logscale) or
                    (visualize_results == 'best' and algorithm.best_state)):
                self.log_data(reporter, epoch, print_log=print_results, alg=alg, argv=argv)

            if save_checkpoint:
                if (any([v in store_networks for v in ['yes', 'last', 'logscale']]) or
                   (iteration+1 == algorithm.n_epochs and store_networks == 'final') or
                   (store_networks == 'all_bests' and algorithm.best_state)):
                    checkpoint_file = self.checkpoints_dir.joinpath(f'checkpoint_{epoch:06d}')
                    algorithm.save_checkpoint(checkpoint_file)

                if algorithm.best_state and store_networks != 'never':
                    checkpoint_file = self.checkpoints_dir.joinpath(f'checkpoint_best')
                    algorithm.save_checkpoint(checkpoint_file)

                if 'last' in store_networks or ('logscale' in store_networks and not logscale):
                    try:
                        self.checkpoints_dir.joinpath(f'checkpoint_{epoch - 1:06d}').unlink()
                    except OSError:
                        pass

    @property
    def snapshot_file(self):
        return self.checkpoints_dir.joinpath(f'checkpoint').str

    def log_data(self, reporter, n, print_log=True, alg=None, argv=None):

        if print_log:
            reporter.print_stats()

        if self.tensorboard_writer is not None:
            logger.info(f"Tensorboard results are stored to: {self.experiment_dir}")
            reporter.write_to_tensorboard(self.tensorboard_writer, hparams=self.tensorboard_hparams)
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
                        write_network(self.tensorboard_writer, net, name, n)
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

        normal_path = [self.hparams.logs_path, self.hparams.project_name,
                       self.hparams.algorithm, self.hparams.identifier]
        pd = path_depth(self.hparams.logs_path)

        return os.path.join(*normal_path[:len(normal_path)-pd-level], path)

    @staticmethod
    def open_tensorboard(root='', project=None, algorithm=None,
                         identifier=None, experiment=None, hparams=False, port=None,
                         get_port_from_beam_port_range=True):
        depth = 4
        filters = {'project': None, 'algorithm': None, 'identifier': None, 'experiment':None}
        if project is not None:
            path_type = check_type(project)
            if path_type.minor == Types.list:
                filters['project'] = project
            else:
                filters['project'] = [project]
                path = os.path.join(root, project)
                if os.path.isdir(path):
                    root = path
                    depth = 3

        if algorithm is not None:
            path_type = check_type(algorithm)
            if path_type.minor == Types.list:
                filters['algorithm'] = algorithm
            else:
                filters['algorithm'] = [algorithm]
                path = os.path.join(root, algorithm)
                if os.path.isdir(path):
                    root = path
                    depth = 2

        if identifier is not None:
            path_type = check_type(identifier)
            if path_type.minor == Types.list:
                filters['identifier'] = identifier
            else:
                filters['identifier'] = [identifier]
                path = os.path.join(root, identifier)
                if os.path.isdir(path):
                    root = path
                    depth = 1
        if experiment is not None:
            path_type = check_type(experiment)
            if path_type.minor == Types.list:
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
            base_dir = os.path.join(self.hparams.logs_path, self.hparams.project_name)
            depth = 3
        elif add_all_of_same_algorithm:
            base_dir = os.path.join(self.hparams.logs_path, self.hparams.project_name, self.hparams.algorithm)
            depth = 2
        elif add_all_of_same_identifier:
            base_dir = os.path.join(self.hparams.logs_path, self.hparams.project_name, self.hparams.algorithm, self.hparams.identifier)
            depth = 1
        else:
            base_dir = self.experiment_dir
            depth = 0

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

    def prepare_experiment_for_run(self):

        logger.set_verbosity(logger.level, file_info=False)

        if self.log_experiment and (not self.load_model and not self.logs_path_is_built):
            self.build_experiment_dir()
        else:
            if not self.log_experiment:
                logger.warning(f"Experiment logs are disabled (log_experiment=False)")
            else:
                logger.debug(f"Experiment setup already exists at {self.experiment_dir}")

    def fit(self, alg=None, dataset=None, algorithm_generator=None, return_results=False, reload_results=False,
            tensorboard_arguments=None, alg_args=None, alg_kwargs=None, dataset_args=None,
            dataset_kwargs=None, runner=None, **kwargs):

        if type(runner) is str and runner == 'simple':
            runner = simple_runner
            if algorithm_generator is None:
                algorithm_generator = simple_algorithm_generator
        elif runner is None or (type(runner) is str and runner == 'default'):
            runner = default_runner
            if algorithm_generator is None:
                algorithm_generator = nn_algorithm_generator

        self.prepare_experiment_for_run()

        try:

            if dataset is not None:
                kwargs['dataset'] = dataset
            if alg_args is not None:
                kwargs['alg_args'] = alg_args
            if alg_kwargs is not None:
                kwargs['alg_kwargs'] = alg_kwargs
            if dataset_args is not None:
                kwargs['dataset_args'] = dataset_args
            if dataset_kwargs is not None:
                kwargs['dataset_kwargs'] = dataset_kwargs
            if tensorboard_arguments is not None:
                kwargs['tensorboard_arguments'] = tensorboard_arguments

            if self.hparams.get('federated_runner'):
                res = self.federated_training(alg=alg, algorithm_generator=algorithm_generator, **kwargs)
            else:
                res = self.run(runner, *(algorithm_generator, self, alg), **kwargs)

        except KeyboardInterrupt:

            res = None
            logger.warning(f"KeyboardInterrupt: Training was interrupted, reloads last checkpoint")

        # take care of what is done after training ends
        if res is None or (self.world_size > 1 and not self.hparams.get('federated_runner')):
            alg = algorithm_generator(self, alg, **kwargs)
            results = None
            self.reload_checkpoint(alg)

            if reload_results:
                results = {}
                for subset in alg.results_dir.iterdir():
                    res = list(subset.iterdir())
                    res = pd.DataFrame({'name': res, 'index': [int(c.name.split('_')[-1]) for c in res]})
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

    def federated_training(self, alg, algorithm_generator=None, dataset=None, alg_args=None, alg_kwargs=None,
                           dataset_args=None, dataset_kwargs=None,
                           ray_kwargs=None, remote_kwargs=None, **kwargs):

        from ..federated import federated_executor, worker_executor

        if algorithm_generator is None:
            algorithm_generator = self.algorithm_generator

        workers = federated_executor(func=worker_executor, world_size=self.world_size,
                                     framework=self.distributed_training_framework,
                                     distributed_backend=self.hparams.get('distributed_backend'),
                                     host=self.hparams.get('mp_ip'), port=self.hparams.get('mp_port'),
                                     kv_store=self.hparams.get('kv_store'),
                                     kv_store_path=self.hparams.get('kv_store_path'),
                                     kv_store_timeout=self.hparams.get('kv_store_timeout'),
                                     kv_store_port=self.hparams.get('kv_store_port'), ray_address=None,
                                     ray_kwargs=ray_kwargs,
                                     num_gpus=self.hparams.get('n_gpus_per_worker'),
                                     num_cpus=self.hparams.get('n_cpus_per_worker'), remote_kwargs=remote_kwargs, **kwargs)

        if self.world_size > 1:
            logger.info(f'Initializing {self.world_size} parallel workers')
        else:
            logger.info(f'Single worker mode')

        results = []
        for i, we in enumerate(workers):
            logger.info(f"Starting worker {i+1}/{len(workers)} on host: {we.hostname.value} "
                        f"with gpus: {we.physical_devices.value}")
            results.append(we(self, alg, algorithm_generator, dataset=dataset, alg_args=alg_args, alg_kwargs=alg_kwargs,
                              dataset_args=dataset_args, dataset_kwargs=dataset_kwargs))

        return results[0].wait()

    @cached_property
    def distributed_training_framework(self):
        if self.training_framework in ['accelerate', 'deepspeed']:
            return 'deepspeed'
        return 'ddp'

    @cached_property
    def default_hparams(self):
        return UniversalConfig(return_defaults=True)

    def build_experiment_dir(self):

        logger.cleanup(clean_default=False)
        logger.add_file_handlers(self.experiment_dir.joinpath('experiment.log'))
        print_beam_hyperparameters(self.hparams, default_params=UniversalConfig(return_defaults=True),
                                   debug_only=not self.print_hyperparameters)

        self.experiment_dir.mkdir()
        self.hparams.to_path(self.experiment_dir.joinpath('hparams.pkl'))

        if not self.hparams.override:
            logger.info(f"Creating new experiment")

        else:
            logger.warning("Deleting old experiment")

            self.experiment_dir.rmtree()
            self.exp_name = "%04d_%s" % (self.exp_num, self.exptime)
            self.experiment_dir = self.experiment_dir.parent.joinpath(self.exp_name)

            # set dirs
            self.tensorboard_dir = self.experiment_dir.joinpath('tensorboard')
            self.checkpoints_dir = self.experiment_dir.joinpath('checkpoints')
            self.results_dir = self.experiment_dir.joinpath('results')
            self.code_dir = self.experiment_dir.joinpath('code')

        logger.info(f"Experiment directory is: {self.experiment_dir}")

        self.tensorboard_dir.joinpath('logs').mkdir()
        self.tensorboard_dir.joinpath('hparams').mkdir()
        self.checkpoints_dir.mkdir()

        # make log dirs
        self.results_dir.mkdir()

        # copy code to dir
        if is_notebook():
            code_root_path = os.getcwd()
        else:
            code_root_path = sys.argv[0]

        self.source_dir = os.path.dirname(os.path.realpath(code_root_path))
        if self.hparams.copy_code:
            self.code_dir.copy(beam_path(self.source_dir), include=['.py', '.md', '.ipynb'])

        self.experiment_dir.joinpath('args.pkl').write(self.vars_args)
        self.logs_path_is_built = True

    def run(self, job, *args, **kwargs):

        arguments = (job, self, *args)

        def _run_mpi(demo_fn, world_size):
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            assert rank == 0, "The main process should be rank 0."

            spawned_comm = []
            for i in range(1, world_size):
                # This does not work as it requires the same executable to be run (not a function)
                spawned_comm.append(MPI.COMM_SELF.Spawn(demo_fn, args=(rank, world_size, kwargs, *arguments),
                                                        maxprocs=world_size))

            # Receive results from child processes
            res = [demo_fn(0, world_size, kwargs, *arguments)]
            for i in range(world_size-1):
                result = spawned_comm[i].recv(source=MPI.ANY_SOURCE)
                res.append(result)
                spawned_comm[i].Disconnect()

            return res

        def _run(demo_fn, world_size):

            ctx = mp.get_context(self.hparams.mp_context)
            results_queue = ctx.Queue()
            for rank in range(world_size):
                ctx.Process(target=demo_fn, args=(rank, world_size, results_queue, *arguments),
                            kwargs=kwargs).start()

            res = []
            for rank in range(world_size):
                res.append(results_queue.get())

            return res

        if self.world_size > 1:
            logger.info(f'Initializing {self.world_size} parallel workers')
            if self.distributed_training_framework == 'ddp':
                logger.warning(f"Caution: Sometimes DDP experiments can fail due to a bad configuration. "
                               f"Specifically, if in_place error set --no-broadcast-buffer flag and for subgraph issues"
                               f"set --find-unused-parameters")

            if self.hparams.mp_port is None:
                self.hparams.set('mp_port', find_free_port())
            elif not check_if_port_is_available(self.hparams.mp_port):
                logger.warning(f"Port {self.hparams.mp_port} is not available. Using random port")
                self.hparams.set('mp_port', find_free_port())

            logger.info(f'Multiprocessing port is: {self.hparams.mp_port}')

            if self.hparams.get('distributed_backend') == 'mpi':
                return _run_mpi(run_worker, self.world_size)
            else:
                return _run(run_worker, self.world_size)
        else:
            logger.info(f'Single worker mode')
            return run_worker(0, 1, None, *arguments, **kwargs)

    def writer_cleanup(self):

        if self.tensorboard_writer is not None:
            self.tensorboard_writer.close()
            self.tensorboard_writer = None

        if self.comet_writer is not None:
            self.comet_writer.close()
            self.comet_writer = None

        if self.mlflow_writer is not None:
            self.mlflow_writer.close()
            self.mlflow_writer = None
