import argparse
import copy
import os
from argparse import Namespace
from dataclasses import dataclass
from .utils import beam_arguments, to_dict
from ..path import beam_path
from .basic_configuration import basic_beam_parser, boolean_feature


@dataclass
class BeamParam:
    name: str
    type: type
    default: any
    help: str
    model: bool = True
    tune: bool = False


class BeamHparams(Namespace):
    parameters = []
    defaults = {}

    def __init__(self, *args, hparams=None, model_set=None, tune_set=None, return_defaults=False, **kwargs):

        if model_set is None:
            model_set = set()
        if tune_set is None:
            tune_set = set()

        if hparams is None:
            parser = basic_beam_parser()

            defaults = None
            parameters = None

            types = type(self).__mro__

            hparam_types = []
            for ti in types:
                if ti is argparse.Namespace:
                    break
                hparam_types.append(ti)

            for ti in hparam_types[::-1]:

                if ti.defaults is not defaults:
                    defaults = ti.defaults
                    d = defaults
                else:
                    d = None

                if ti.parameters is not parameters:
                    parameters = ti.parameters
                    h = parameters
                else:
                    h = None

                ms, ts = self.update_parser(parser, defaults=d, parameters=h)
                tune_set = tune_set.union(ts)
                model_set = model_set.union(ms)

            if return_defaults:
                hparams = parser.parse_args([])
            else:
                hparams = beam_arguments(parser, *args, **kwargs)
            tune_set = tune_set.union(hparams.tune_set)
            model_set = model_set.union(hparams.model_set)

            del hparams.tune_set
            del hparams.model_set

        elif isinstance(hparams, BeamHparams):
            tune_set = tune_set.union(hparams.tune_parameters.__dict__.keys())
            model_set = model_set.union(hparams.model_parameters.__dict__.keys())

        elif isinstance(hparams, Namespace):

            if hasattr(hparams, 'tune_set'):
                tune_set = tune_set.union(hparams.tune_set)
                del hparams.tune_set

            if hasattr(hparams, 'model_set'):
                model_set = model_set.union(hparams.model_set)
                del hparams.model_set
            else:
                model_set = model_set.union(hparams.__dict__.keys())

        elif isinstance(hparams, dict):
            model_set = model_set.union(hparams.keys())

        else:
            raise ValueError(f"Invalid hparams type: {type(hparams)}")

        self._model_set = model_set
        self._tune_set = tune_set

        super().__init__(**hparams.__dict__)

    @classmethod
    def defaults(cls):
        return cls(return_defaults=True)

    @classmethod
    def add_argument(cls, name, type, default, help, model=True, tune=False):
        cls.parameters.append(BeamParam(name, type, default, help, model, tune))

    @classmethod
    def add_arguments(cls, *args):
        for arg in args:
            cls.add_argument(**arg)

    @classmethod
    def remove_argument(cls, name):
        cls.parameters = [p for p in cls.parameters if p.name != name]

    @classmethod
    def remove_arguments(cls, *args):
        for arg in args:
            cls.remove_argument(arg)

    @classmethod
    def set_defaults(cls, **kwargs):
        cls.defaults.update(kwargs)

    @classmethod
    def set_default(cls, name, value):
        cls.defaults[name] = value

    def dict(self):
        return to_dict(self)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return (f"{type(self).__name__}:\n\nSystem Parameters:\n\n{self.system_parameters}\n\n"
                f"Model Parameters:\n\n{self.model_parameters}\n\nTune Parameters:\n\n{self.tune_parameters}")

    @property
    def tune_parameters(self):
        return Namespace(**{k: getattr(self, k) for k in self._tune_set})

    @property
    def model_parameters(self):
        return Namespace(**{k: getattr(self, k) for k in self._model_set})

    def items(self):
        for k, v in vars(self).items():
            if k.startswith('_'):
                continue
            yield k, v

    def keys(self):
        for k in vars(self).keys():
            if k.startswith('_'):
                continue
            yield k

    def values(self):
        for k, v in self.items():
            yield v

    @property
    def system_parameters(self):
        return Namespace(**{k: v for k, v in self.items() if k not in self._tune_set.union(self._model_set)})

    @staticmethod
    def update_parser(parser, defaults=None, parameters=None):

        model_set = set()
        tune_set = set()

        if defaults is not None:
            # set defaults
            parser.set_defaults(**{k.replace('-', '_'): v for k, v in defaults.items()})

        if parameters is not None:
            for v in parameters:

                name_to_parse = v.name.replace('_', '-')
                name_to_store = v.name.replace('-', '_')

                if v.model:
                    model_set.add(name_to_store)

                if v.tune:
                    tune_set.add(name_to_store)

                if v.type is bool:
                    boolean_feature(parser, name_to_parse, v.default, v.help)
                elif v.type is list:
                    parser.add_argument(f"--{name_to_parse}", type=v.type, default=v.default, nargs='+', help=v.help)
                else:
                    parser.add_argument(f"--{name_to_parse}", type=v.type, default=v.default, help=v.help)

        return model_set, tune_set

    def to_path(self, path, ext=None):
        d = copy.deepcopy(self.dict())
        d['_model_set'] = list(self._model_set)
        d['_tune_set'] = list(self._tune_set)
        beam_path(path).write(d, ext=ext)

    @classmethod
    def from_path(cls, path, ext=None):
        d = beam_path(path).read(ext=ext)
        model_set = set(d.pop('_model_set', set(d.keys())))
        tune_set = set(d.pop('_tune_set', set()))
        return cls(hparams=d, model_set=model_set, tune_set=tune_set)

    def is_hparam(self, key):
        key = key.replace('-', '_')
        if key in self.hparams:
            return True
        return False

    def __getitem__(self, item):
        item = item.replace('-', '_')
        r = getattr(self, item)
        if r is None and item in os.environ:
            r = os.environ[item]
        return r

    def __setitem__(self, key, value):
        key = key.replace('-', '_')
        setattr(self, key, value)

    def get(self, hparam, default=None, preferred=None, specific=None):

        if preferred is not None:
            return preferred

        if type(specific) is list:
            for s in specific:
                if f"{hparam}_{s}" in self:
                    return getattr(self, f"{specific}_{hparam}")
        elif specific is not None and f"{specific}_{hparam}" in self:
            return getattr(self, f"{specific}_{hparam}")

        if hparam in self:
            return getattr(self, hparam)

        return default
