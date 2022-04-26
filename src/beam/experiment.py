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
from .utils import include_patterns, logger, set_seed
import pandas as pd
import torch.multiprocessing as mp
from .utils import setup, cleanup, set_seed
import torch.distributed as dist

# check

def run_worker(rank, world_size, job, experiment, *args):

    if world_size > 1:
        setup(rank, world_size)

    experiment.set_rank(rank, world_size)
    set_seed(seed=experiment.seed, constant=rank, increment=False, deterministic=experiment.deterministic)
    job(rank, world_size, experiment, *args)

    if world_size > 1:
        cleanup(rank, world_size)


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

    def __init__(self, args, results_names=None):
        """
        args: the parsed arguments
        results_names: additional results directories (defaults are: train, validation, test)
        """

        set_seed(args.seed)

        # torch.set_num_threads(100)
        logger.info(f"beam project: {args.project_name}")
        logger.info('Simulation Hyperparameters')

        self.args = vars(args)
        for k, v in vars(args).items():
            logger.info(k + ': ' + str(v))
            setattr(self, k, v)

        # determine the batch size

        # parameters

        self.start_time = time.time()
        self.exptime = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self.device = torch.device(int(self.device) if x.isnumeric() else self.device)
        self.base_dir = os.path.join(self.root_dir, self.project_name)

        for folder in [self.base_dir, self.root_dir]:
            if not os.path.exists(folder):
                os.makedirs(folder)

        dirs = os.listdir(self.base_dir)
        temp_name = "%s_%s_exp" % (self.algorithm, self.identifier)
        self.exp_name = None
        self.load_model = False

        ds = [d for d in dirs if temp_name in d]
        ns = np.array([int(d.split("_")[-3]) for d in ds])

        if self.reload:

            if self.resume >= 0:
                for d in dirs:
                    if "%s_%04d_" % (temp_name, self.resume) in d:
                        self.exp_name = d
                        self.exp_num = self.resume
                        self.load_model = True
                        break
            else:
                if len(ns):
                    n_max = np.argmax(ns)
                    self.exp_name = ds[n_max]
                    self.exp_num = n_max
                    self.load_model = True

        else:

            if self.override:

                self.exp_num = np.argmax(ns) if len(ns) else 0
                self.load_model = bool(len(ns))

                if self.load_model:
                    for d in dirs:
                        if "%s_%04d_" % (temp_name, self.exp_num) in d:
                            self.exp_name = d
                            break

        if self.exp_name is None:
            self.exp_num = np.argmax(ns) + 1 if len(ns) else 0
            self.exp_name = "%s_%04d_%s" % (temp_name, self.exp_num, self.exptime)

        # init experiment parameters
        self.root = os.path.join(self.base_dir, self.exp_name)

        # set dirs
        self.tensorboard_dir = os.path.join(self.root, 'tensorboard')
        self.checkpoints_dir = os.path.join(self.root, 'checkpoints')
        self.results_dir = os.path.join(self.root, 'results')
        self.code_dir = os.path.join(self.root, 'code')

        if self.load_model and self.reload:
            logger.info("Resuming existing experiment")

        else:

            if not self.load_model:
                logger.info("Creating new experiment")

            else:
                logger.info("Deleting old experiment")
                shutil.rmtree(self.root)

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

            # copy code to dir
            copytree(os.path.dirname(os.path.realpath(__file__)), self.code_dir,
                     ignore=include_patterns('*.py', '*.md', '*.ipynb'))

            pd.to_pickle(vars(args), os.path.join(self.root, "args.pkl"))

        self.epoch = 0
        self.writer = None
        self.rank = 0
        self.world_size = self.parallel

        if self.world_size > 1:
            torch.multiprocessing.set_sharing_strategy('file_system')

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


    def set_rank(self, rank, world_size):

        self.rank = rank
        self.world_size = world_size

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
                logger.info(f'Finished epoch {self.epoch}:')

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

            for param, val in res['scalar'].items():
                if type(val) is dict or type(val) is defaultdict:
                    for p, v in val.items():
                        val[p] = np.mean(v)
                elif isinstance(res['scalar'][param], torch.Tensor):
                    res['scalar'][param] = torch.mean(val)
                else:
                    res['scalar'][param] = np.mean(val)


            if print_log:
                logger.info(f'{subset}:')
                for param in res['scalar']:
                    if not (type(res['scalar'][param]) is dict or type(
                            res['scalar'][param]) is defaultdict):
                        logger.info('%s %g \t|' % (param, res['scalar'][param]))

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

            for log_type in res:
                log_func = getattr(self.writer, f'add_{log_type}')
                for param in res[log_type]:
                    if type(res[log_type][param]) is dict or type(res[log_type][param]) is defaultdict:
                        for p, v in res[log_type][param].items():
                            log_func(f'{subset}_{param}/{p}', v, n, **defaults_argv[log_type][param])
                    elif type(res[log_type][param]) is list:
                        log_func(f'{subset}/{param}', *res[log_type][param], n, **defaults_argv[log_type][param])
                    else:
                        log_func(f'{subset}/{param}', res[log_type][param], n, **defaults_argv[log_type][param])


    def run(self, job, *args):

        arguments = (job, self, *args)

        def _run(demo_fn, world_size):
            mp.spawn(demo_fn,
                     args=(world_size, *arguments),
                     nprocs=world_size,
                     join=True)

        if self.parallel > 1:
            _run(run_worker, self.parallel)
        else:
            run_worker(0, 1, *arguments)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.tensorboard:
            self.writer.export_scalars_to_json(os.path.join(self.tensorboard_dir, "all_scalars.json"))
            self.writer.close()
