import time
import numpy as np
import os
import torch
from collections import defaultdict
import pandas as pd
from contextlib import contextmanager
from timeit import default_timer as timer
import threading
from ..utils import cached_property

from ..utils import (pretty_format_number, as_numpy, pretty_print_timedelta, recursive_flatten, rate_string_format,
                     nested_defaultdict, as_tensor, squeeze_scalar, check_type, check_element_type,
                     strip_prefix, recursive_detach, recursive_to_cpu)

from ..utils import tqdm_beam as tqdm
from ..logging import beam_logger as logger
from ..data import BeamData
from ..type import Types


class BeamReport(object):

    def __init__(self, objective=None, optimization_mode='max', aux_objectives=None, aux_objectives_modes=None):

        self.scalar = None
        self.aux = None

        self.scalars = None
        self.buffer = None

        self.scalar_kwargs = None
        self.scalars_kwargs = None
        self.aux_kwargs = None

        self.scalar_aggregation = None
        self.scalars_aggregation = None

        if aux_objectives is None:
            aux_objectives = []
        if aux_objectives_modes is None:
            aux_objectives_modes = []

        if objective is not None:
            aux_objectives.insert(0, objective)
            aux_objectives_modes.insert(0, optimization_mode)

        self.objective_names = aux_objectives
        self.objectives_modes = aux_objectives_modes

        self.objective_name = None
        self.optimization_mode = None

        self.epoch = None
        self.best_epoch = None
        self.objective = None
        self.best_objective = None
        self.subset_context = None
        self.best_state = False
        self.iteration = None
        self._data = None
        self.state = None
        self.t0 = time.time()
        self.n_epochs = None
        self.total_time = None
        self.estimated_time = None
        self.first_epoch = None
        self.subsets_keys = None
        self.batch_size_context = None
        self.total_epochs = None
        self.stack = []
        self.stack_size = 4

    def info(self, msg, new=False):
        logger.info(msg)
        if new:
            self.stack.append(msg)
        else:
            self.stack[-1] = f"{self.stack[-1]}\n{msg}"
        self.stack = self.stack[-self.stack_size:]

    def llm_info(self):
        logs = "\n\n".join(self.stack)
        prompt = f"You are an ML expert. You need to interpret and analyze logs of deep learning training runs. \n" \
                 f"The logs contains metrics and reports of the training process. \n" \
                 f"Please provide an analysis and suggest solutions for any problems you find. For example: \n" \
                 f"overfitting, underfitting, etc. Be concise and respond in less than 100 words \n" \
                 f"========================================================================\n\n" \
                 f"These are the experiment logs: \n\n" \
                 f"{logs}\n\n" \
                 f"Response: \"\"\"\n{{text input here}}\n\"\"\""

        llm_response = self.llm.ask(prompt)
        print()
        logger.info(f"LLM response: {llm_response.text}")

    def reset_epoch(self, epoch, total_epochs=None):

        self._data = None
        self.state = 'before_epoch'
        self.scalar = defaultdict(list)
        self.aux = defaultdict(dict)

        self.scalars = nested_defaultdict(list)
        self.buffer = nested_defaultdict(list)

        self.scalar_kwargs = {}
        self.scalars_kwargs = {}
        self.aux_kwargs = defaultdict(dict)

        self.scalar_aggregation = {}
        self.scalars_aggregation = {}

        self.subsets_keys = nested_defaultdict(list)
        self.epoch = epoch
        if total_epochs is None:
            total_epochs = epoch + 1
        self.total_epochs = total_epochs

    def reset_time(self, first_epoch=None, n_epochs=None):
        self.t0 = time.time()
        self.n_epochs = n_epochs
        if first_epoch is None:
            first_epoch = 0
        self.first_epoch = first_epoch
        self.estimated_time = None
        self.total_time = None

    @property
    def data(self):
        if self._data is None:
            data = nested_defaultdict(dict)
            for k, v in self.scalar.items():
                subset, name = self.extract_subset_and_name(k)
                if subset is None:
                    data['scalar'][name] = v
                else:
                    data[subset]['scalar'][name] = v
            for k, v in self.scalars.items():
                subset, name = self.extract_subset_and_name(k)
                if subset is None:
                    data['scalars'][name] = v
                else:
                    data[subset]['scalars'][name] = v
            for dtype, v_dtype in self.aux.items():
                for k, v in v_dtype.items():
                    subset, name = self.extract_subset_and_name(k)
                    if subset is None:
                        data[dtype][name] = v
                    else:
                        data[subset][dtype][name] = v

            data['global'] = {'epoch': self.epoch,
                              'best_epoch': self.best_epoch,
                              'objective': self.objective,
                              'best_objective': self.best_objective,
                              'total_time': self.total_time,
                              'estimated_time': self.estimated_time,
                              }

            data['objective'] = self.objective

            self._data = BeamData(data)

        return self._data

    def write_to_path(self, path):
        self.data.store(path=path)

    @staticmethod
    def format_stat(k, v):
        format = f"{k if k != 'mean' else 'avg'}:{pretty_format_number(v)}".ljust(15)
        return format

    @staticmethod
    def format(v):
        v_type = check_element_type(v)
        if v_type == Types.int:
            if v >= 1000:
                return f"{float(v): .4}"
            else:
                return str(v)
        elif v_type == Types.float:
            return f"{v: .4}"
        else:
            return v

    @cached_property
    def llm(self):
        from ..config import get_beam_llm
        return get_beam_llm()

    def print_stats(self):

        for subset, data_keys in self.subsets_keys.items():

            self.info(f'{subset}:')

            if 'stats' in data_keys:
                stats = data_keys['stats']
                self.info('| '.join([f"{k}: {BeamReport.format(self.aux['stats'][f'{subset}/{k}'])} " for k in stats]))

            if Types.scalar in data_keys:
                scalars = data_keys['scalar']
                for k in scalars:
                    v = self.scalar[f'{subset}/{k}']

                    v_type = check_type(v)
                    if v_type.major != Types.scalar and np.var(as_numpy(v)) > 0:
                        stat = pd.Series(v, dtype=np.float32).describe()
                    else:
                        if v_type.major != Types.scalar:
                            v = v[0]
                        v = int(v) if v_type.element == Types.int else float(v)
                        stat = {'val': v}

                    stat = '| '.join([BeamReport.format_stat(k, v) for k, v in dict(stat).items() if k != 'count'])

                    if len(k) > 11:
                        paramp = f'{k[:4]}...{k[-4:]}:'
                    else:
                        paramp = f'{k}:'

                    self.info(f'{paramp: <12} | {stat}')

        if self.llm is not None:
            threading.Thread(target=self.llm_info).start()

    def print_metadata(self):

        self.info('----------------------------------------------------------'
                    '---------------------------------------------------------------------', new=not self.best_state)
        objective_str = ''
        if self.best_objective is not None:
            objective_str = f"Current objective: {pretty_format_number(self.objective)} " \
                            f"(Best objective: {pretty_format_number(self.best_objective)} " \
                            f" at epoch {self.best_epoch + 1})"

        if self.epoch is not None:
            done_epochs = self.epoch - self.first_epoch
            self.info(f'Finished epoch {done_epochs + 1}/{self.n_epochs} (Total trained epochs {self.total_epochs}). '
                        f'{objective_str}')

        if self.total_time is not None:
            total_time = pretty_print_timedelta(self.total_time)
            if self.estimated_time:
                estimated_time = pretty_print_timedelta(self.estimated_time)
            else:
                estimated_time = 'N/A'
            self.info(f'Elapsed time: {total_time}. Estimated remaining time: {estimated_time}.', )

    def write_to_tensorboard(self, writer, hparams=None):

        metrics = {}
        for k, v in self.scalar.items():
            kwargs = self.scalar_kwargs.get(k, {})
            agg = self.scalar_aggregation.get(k, 'mean')
            v = self.aggregate_scalar(v, agg)
            if agg != 'mean':
                k = f'{k}_{agg}'
            writer.add_scalar(k, v, **kwargs)
            metrics[k] = v

        if hparams is not None and len(metrics):
            writer.add_hparams(hparams, metrics, name=os.path.join('..', 'hparams'), global_step=self.epoch)

        for k, v in self.scalars.items():

            v_agg = {}
            kwargs = self.scalars_kwargs.get(k, {})
            agg = self.scalars_aggregation.get(k, 'mean')

            for kk, vv in v.items():
                v_agg[kk] = self.aggregate_scalar(vv, agg)

            writer.add_scalars(k, v_agg, **kwargs)

        for k, v in self.aux.items():
            if hasattr(writer, f'add_{k}'):
                writer_func = getattr(writer, f'add_{k}')
                for kk, vv in v.items():
                    kwargs = self.aux_kwargs.get(k, {}).get(kk, {})
                    writer_func(kk, vv, **kwargs)

    @staticmethod
    def extract_subset_and_name(k):
        if '/' not in k:
            subset = None
            name = k
        else:
            subset = k.split('/')[0]
            name = strip_prefix(k, f"{subset}/")
        return subset, name

    @cached_property
    def comparison(self):
        return {'max': np.greater, 'min': np.less}[self.optimization_mode]

    def set_objective(self, objective):

        self.objective = objective

        if self.best_objective is None or self.comparison(self.objective, self.best_objective):
            self.info(f"Epoch {self.epoch+1}: The new best objective is {pretty_format_number(objective)}", new=True)
            self.best_objective = objective
            self.best_epoch = self.epoch
            self.best_state = True
        else:
            self.best_state = False

    def set_objective_name(self, keys):

        for i, o in enumerate(self.objective_names):
            if o is not None and o in keys:
                self.objective_name = o
                self.optimization_mode = self.objectives_modes[i]
                return

    def pre_epoch(self, subset, batch_size=None):
        self.subset_context = subset
        self.batch_size_context = batch_size
        self.state = 'in_epoch'
        return timer()

    def post_epoch(self, subset, t0, batch_size=None, track_objective=True):

        delta = timer() - t0
        n_iter = self.iteration + 1
        batch_size = batch_size or 1

        self.state = 'after_epoch'

        self.report_data('seconds', delta, data_type='stats')
        self.report_data('batches', n_iter, data_type='stats')
        self.report_data('samples', n_iter * batch_size, data_type='stats')
        self.report_data('batch_rate', rate_string_format(n_iter, delta), data_type='stats')
        self.report_data('sample_rate', rate_string_format(n_iter * batch_size, delta), data_type='stats')

        self.total_time = time.time() - self.t0
        if (self.n_epochs is not None) and (self.n_epochs is not None) and (self.n_epochs > 0):
            n_epochs = self.n_epochs - self.first_epoch
            epoch = self.epoch - self.first_epoch

            if epoch + 1 > 0:
                self.estimated_time = self.total_time * (n_epochs - epoch - 1) / (epoch + 1)
            else:
                self.estimated_time = None

        agg = None

        if self.objective_name is None:
            self.set_objective_name(list(self.subsets_keys[subset]['scalar']))

        for name in self.subsets_keys[subset]['scalar']:

            k = f'{subset}/{name}' if subset is not None else name
            v = self.scalar[k]

            self.scalar[k] = self.stack_scalar(v, batch_size=batch_size)

            if name == self.objective_name and track_objective:
                agg = self.scalar_aggregation.get(k, None)
                self.set_objective(self.aggregate_scalar(self.scalar[k], agg))

        for data_type in self.subsets_keys[subset]:
            if data_type in ['scalar', 'scalars', 'stats']:
                continue
            for name in self.subsets_keys[subset][data_type]:
                k = f'{subset}/{name}' if subset is not None else name
                v = self.aux[data_type][k]
                self.aux[data_type][k] = recursive_to_cpu(v)

        if self.objective_name and track_objective and agg is None:
            logger.warning(f"The objective {self.objective_name} is missing from the validation results")

        for name in self.subsets_keys[subset]['scalars']:
            k = f'{subset}/{name}' if subset is not None else name
            v = self.scalar[k]
            for kk, vv in v.items():
                self.scalars[k][kk] = self.stack_scalar(vv, batch_size=batch_size)

        self.subset_context = None
        self.batch_size_context = None

    @contextmanager
    def track_epoch(self, subset, batch_size=None, training=True):
        t0 = self.pre_epoch(subset, batch_size)
        yield
        self.post_epoch(subset, t0, batch_size, not training)

    def iterate(self, generator, **kwargs):
        for i, batch in tqdm(generator, **kwargs):
            self.iteration = i
            yield i, batch

    def set_iteration(self, i):
        self.iteration = i

    @staticmethod
    def detach_scalar(val):
        recursive_detach(val)
        return val

    # @staticmethod
    # def detach_scalar(val):
    #     val_type = check_type(val)
    #     if val_type.major == Types.scalar:
    #         if val_type.element == Types.float:
    #             val = float(val)
    #             pass
    #         elif val_type.element == Types.int:
    #             val = int(val)
    #     elif val_type.minor == Types.tensor:
    #         val = val.detach().cpu()
    #     elif val_type.major == Types.container:
    #         val = as_tensor(val, device='cpu')
    #
    #     return val

    @staticmethod
    def stack_scalar(val, batch_size=None):

        val_type = check_type(val)

        if val_type.major == Types.container and val_type.minor == Types.list:

            v_minor = check_type(val[0]).minor
            if v_minor == Types.tensor:
                oprs = {'cat': torch.cat, 'stack': torch.stack}
            elif v_minor == Types.numpy:
                oprs = {'cat': np.concatenate, 'stack': np.stack}
            elif v_minor == Types.pandas:
                oprs = {'cat': pd.concat, 'stack': pd.concat}
            elif v_minor == Types.native:
                oprs = {'cat': torch.tensor, 'stack': torch.tensor}
            elif v_minor == Types.cudf:
                import cudf
                oprs = {'cat': cudf.concat, 'stack': cudf.concat}
            elif v_minor == Types.polars:
                import polars as pl
                oprs = {'cat': pl.concat, 'stack': pl.concat}
            else:
                oprs = {'cat': lambda x: x, 'stack': lambda x: x}

            opr = oprs['cat']
            if batch_size is not None and hasattr(val[0], '__len__') \
                    and len(val[0]) != batch_size and len(val[0]) == len(val[-1]):
                opr = oprs['stack']

            val = opr(val)

        elif val_type.major == Types.array and val_type.minor == Types.list and val_type.element in [Types.int, Types.float]:
            val = as_tensor(val, device='cpu')

        val = squeeze_scalar(val)

        return val

    @staticmethod
    def aggregate_scalar(val, aggregation, batch_size=None):

        val = BeamReport.stack_scalar(val, batch_size=batch_size)

        if aggregation is None:
            return val

        agg_dict = {'tensor': {'mean': torch.mean, 'sum': torch.sum, 'max': torch.max, 'min': torch.min, 'std': torch.std,
                    'median': torch.median, 'var': torch.var},
                    'numpy': {'mean': np.mean, 'sum': np.sum, 'max': np.max, 'min': np.min, 'std': np.std,
                                'median': np.median, 'var': np.var},
                    }

        val = recursive_flatten(val, flat_array=True, tolist=False)
        val_type = check_type(val)
        if val_type.major == Types.scalar:
            val = float(val)
        elif val_type.minor in [Types.pandas, Types.cudf]:
            val = val.values
            val = agg_dict['numpy'][aggregation](val)
            if val_type.minor == Types.cudf:
                val = float(val)
        elif val_type.minor == Types.polars:
            val = val.to_numpy()
            val = agg_dict['numpy'][aggregation](val)
        elif val_type.minor == Types.tensor:
            val = agg_dict['tensor'][aggregation](val)
        else:
            val = agg_dict['numpy'][aggregation](val)

        val = squeeze_scalar(val)

        return val

    def get_scalar(self, name, subset=None, aggregate=False, stack=False, index=None):
        if subset is None:
            subset = self.subset_context
        k = name if subset is None else f'{subset}/{name}'

        if k in self.scalar:
            v = self.scalar[k]
            if aggregate:
                agg = self.scalar_aggregation.get(f'{subset}/{name}', 'mean')
                v = self.aggregate_scalar(v, agg, batch_size=self.batch_size_context)
            elif index is not None:
                v = v[index]
            elif stack:
                v = self.stack_scalar(v, batch_size=self.batch_size_context)
            return v
        return None

    def get_scalars(self, name, subset=None, aggregate=False):
        if subset is None:
            subset = self.subset_context

        key = name if subset is None else f'{subset}/{name}'
        if key in self.scalars:
            if aggregate:
                agg = self.scalar_aggregation.get(key, 'mean')
                val_dict = self.scalars[key]
                val_dict_agg = {}

                for k, val in val_dict.items():
                    val_dict_agg[k] = self.aggregate_scalar(val, agg, batch_size=self.batch_size_context)

                return val_dict_agg
            return self.scalars[key]
        return None

    def get_data(self, name, data_type=None, subset=None):
        if subset is None:
            subset = self.subset_context
        key = name if subset is None else f'{subset}/{name}'
        if key in self.data[data_type]:
            return self.data[data_type][key]
        return None

    def get_buffer(self, name, subset=None):
        if subset is None:
            subset = self.subset_context
        key = name if subset is None else f'{subset}/{name}'
        if key in self.buffer:
            return self.buffer[key]
        return None

    def get_image(self, name, subset=None):
        return self.get_data(name, data_type='image', subset=subset)

    def get_images(self, name, subset=None):
        return self.get_data(name, data_type='images', subset=subset)

    def get_histogram(self, name, subset=None):
        return self.get_data(name, data_type='histogram', subset=subset)

    def get_figure(self, name, subset=None):
        return self.get_data(name, data_type='figure', subset=subset)

    def get_audio(self, name, subset=None):
        return self.get_data(name, data_type='audio', subset=subset)

    def get_video(self, name, subset=None):
        return self.get_data(name, data_type='video', subset=subset)

    def get_text(self, name, subset=None):
        return self.get_data(name, data_type='text', subset=subset)

    def get_embedding(self, name, subset=None):
        return self.get_data(name, data_type='embedding', subset=subset)

    def get_mesh(self, name, subset=None):
        return self.get_data(name, data_type='mesh', subset=subset)

    def get_pr_curve(self, name, subset=None):
        return self.get_data(name, data_type='pr_curve', subset=subset)

    def report_scalar(self, name, val, subset=None, aggregation=None, append=None, **kwargs):

        if subset is None:
            subset = self.subset_context

        key = name if subset is None else f'{subset}/{name}'
        if append is None:
            append = self.state == 'in_epoch'

        val = self.detach_scalar(val)

        if append:
            self.scalar[key].append(val)
        else:
            self.scalar[key] = val

        kwargs['global_step'] = self.epoch
        self.scalar_kwargs[key] = kwargs

        if name not in self.subsets_keys[subset]['scalar']:
            self.subsets_keys[subset]['scalar'].append(name)

        if aggregation is None:
            aggregation = 'mean'
        self.scalar_aggregation[key] = aggregation

    def report_scalars(self, name, val_dict, subset=None, aggregation=None, append=None, **kwargs):

        if subset is None:
            subset = self.subset_context

        key = name if subset is None else f'{subset}/{name}'
        if append is None:
            append = self.state == 'in_epoch'

        for k, val in val_dict.items():

            val = self.detach_scalar(val)
            if append:
                self.scalars[key][k].append(val)
            else:
                self.scalars[key][k] = val

        kwargs['global_step'] = self.epoch
        self.scalar_kwargs[key] = kwargs

        if name not in self.subsets_keys[subset]['scalars']:
            self.subsets_keys[subset]['scalars'].append(name)

        if aggregation is None:
            aggregation = 'mean'
        self.scalars_aggregation[key] = aggregation

    def report_data(self, name, val, subset=None, data_type=None, **kwargs):

        if subset is None:
            subset = self.subset_context

        key = name if subset is None else f'{subset}/{name}'
        if data_type is None:
            data_type = 'other'

        self.aux[data_type][key] = val

        if name not in self.subsets_keys[subset][data_type]:
            self.subsets_keys[subset][data_type].append(name)

        kwargs['global_step'] = self.epoch
        self.aux_kwargs[data_type][key] = kwargs

    def report_histogram(self, name, val, subset=None, **kwargs):
        self.report_data(name, val, subset, 'histogram', **kwargs)

    def report_image(self, name, val, subset=None, **kwargs):
        self.report_data(name, val, subset, 'image', **kwargs)

    def report_images(self, name, val, subset=None, **kwargs):
        self.report_data(name, val, subset, 'images', **kwargs)

    def report_figure(self, name, val, subset=None, **kwargs):
        self.report_data(name, val, subset, 'figure', **kwargs)

    def report_video(self, name, val, subset=None, **kwargs):
        self.report_data(name, val, subset, 'video', **kwargs)

    def report_audio(self, name, val, subset=None, **kwargs):
        self.report_data(name, val, subset, 'audio', **kwargs)

    def report_text(self, name, val, subset=None, **kwargs):
        self.report_data(name, val, subset, 'text', **kwargs)

    def report_embedding(self, name, val, subset=None, **kwargs):
        self.report_data(name, val, subset, 'embedding', **kwargs)

    def report_mesh(self, name, val, subset=None, **kwargs):
        self.report_data(name, val, subset, 'mesh', **kwargs)

    def report_pr_curve(self, name, labels, predictions, subset=None, **kwargs):
        kwargs['predictions'] = predictions
        self.report_data(name, labels, subset, 'pr_curve', **kwargs)

    def add_buffer(self, name, val, subset=None):
        if subset is None:
            subset = self.subset_context

        if name not in self.subsets_keys[subset]['buffer']:
            self.subsets_keys[subset]['buffer'].append(name)

        if subset is None:
            self.buffer[name].append(val)
        else:
            self.buffer[subset][name].append(val)
