import argparse
import copy
import os
from argparse import Namespace
from dataclasses import dataclass
from .utils import beam_arguments, to_dict
from ..path import beam_path
from .basic_configuration import basic_beam_parser, empty_beam_parser, boolean_feature


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
    use_basic_parser = True

    def __init__(self, *args, hparams=None, model_set=None, tune_set=None, return_defaults=False, **kwargs):

        if model_set is None:
            model_set = set()
        if tune_set is None:
            tune_set = set()

        if hparams is None:

            t = type(self)
            if t.use_basic_parser:
                parser = basic_beam_parser()
            else:
                parser = empty_beam_parser()

            defaults = None
            parameters = None

            types = type(self).__mro__

            hparam_types = []
            for ti in types:
                if not issubclass(ti, argparse.Namespace) or ti is argparse.Namespace:
                    continue
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

                parser, ms, ts = self.update_parser(parser, defaults=d, parameters=h)
                tune_set = tune_set.union(ts)
                model_set = model_set.union(ms)

            hparams = beam_arguments(parser, *args, return_defaults=return_defaults, **kwargs)
            tune_set = tune_set.union(hparams.tune_set)
            model_set = model_set.union(hparams.model_set)

            del hparams.tune_set
            del hparams.model_set

            hparams = hparams.__dict__

        elif isinstance(hparams, BeamHparams):
            tune_set = tune_set.union(hparams.tune_parameters.__dict__.keys())
            model_set = model_set.union(hparams.model_parameters.__dict__.keys())

            hparams = hparams.__dict__

        elif isinstance(hparams, dict) or isinstance(hparams, Namespace):

            if isinstance(hparams, Namespace):
                hparams = vars(hparams)
            hparams = copy.deepcopy(hparams)

            if 'tune_set' in hparams:
                tune_set = tune_set.union(hparams.pop('_tune_set'))

            if 'model_set' in hparams:
                model_set = model_set.union(hparams.pop('_model_set'))
            else:
                model_set = model_set.union(hparams.keys())

        else:
            raise ValueError(f"Invalid hparams type: {type(hparams)}")

        self._model_set = model_set
        self._tune_set = tune_set

        super().__init__(**hparams)

    @classmethod
    def default_values(cls):
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

    @property
    def namespace(self):
        return Namespace(**self.__dict__)

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
            parser.set_defaults(**{k.replace('-', '_').strip(): v for k, v in defaults.items()})

        if parameters is not None:
            for v in parameters:

                name_to_parse = v.name.replace('_', '-').strip()
                name_to_store = v.name.replace('-', '_').strip()

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

        return parser, model_set, tune_set

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
        key = key.replace('-', '_').strip()
        if key in self.hparams:
            return True
        return False

    def __getitem__(self, item):
        item = item.replace('-', '_').strip()
        r = getattr(self, item)
        if r is None and item in os.environ:
            r = os.environ[item]
        return r

    def __setitem__(self, key, value):
        self.set(key, value)

    def update(self, hparams, tune=None, model=None):
        for k, v in hparams.items():
            self.set(k, v, tune=tune, model=model)

    def set(self, key, value, tune=None, model=None):
        key = key.replace('-', '_').strip()
        if key in self.__dict__:
            if key in self._model_set and model is not None:
                if key in self._model_set:
                    self._model_set.remove(key)
            if key in self._tune_set and tune is not None:
                if key in self._tune_set:
                    self._tune_set.remove(key)
        setattr(self, key, value)
        if tune:
            self._tune_set.add(key)
        if model is not None and not model:
            if key in self._model_set:
                self._model_set.remove(key)

    def __setattr__(self, key, value):
        key = key.replace('-', '_').strip()
        if key.startswith('_'):
            super().__setattr__(key, value)
        else:
            if key not in self.__dict__:  # new key
                self._model_set.add(key)
            super().__setattr__(key, value)

    def get(self, key, default=None, preferred=None, specific=None):

        key = key.replace('-', '_').strip()
        if preferred is not None:
            return preferred

        if type(specific) is list:
            for s in specific:
                if f"{key}_{s}" in self:
                    return getattr(self, f"{specific}_{key}")
        elif specific is not None and f"{specific}_{key}" in self:
            return getattr(self, f"{specific}_{key}")

        if key in self:
            return getattr(self, key)

        return default
