import argparse
import copy
import os
from argparse import Namespace
from collections import defaultdict
from typing import List, Union
from pprint import pformat

from dataclasses import dataclass, field
from .utils import to_dict, empty_beam_parser, boolean_feature, _beam_arguments
from ..path import beam_path
import json


@dataclass
class BeamParam:
    name: str
    type: type
    default: any
    help: str
    tags: Union[List[str], str, None] = None


class BeamConfig(Namespace):
    parameters = []
    defaults = {}

    def __init__(self, *args, config=None, tags=None, return_defaults=False, silent=False, strict=False, **kwargs):

        if tags is None:
            tags = defaultdict(set)
        elif isinstance(tags, dict):
            for k, v in tags.items():
                if isinstance(v, str):
                    tags[k] = {v}
                else:
                    tags[k] = set(v)
            tags = defaultdict(set, tags)

        if config is None:

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

                parser = self.update_parser(parser, defaults=d, parameters=h, source=ti.__name__)

            config, more_tags = _beam_arguments(parser, *args, return_defaults=return_defaults,
                                                return_tags=True, silent=silent, strict=strict, **kwargs)

            for k, v in more_tags.items():
                tags[k] = tags[k].union(v)

            config = config.__dict__

        elif isinstance(config, BeamConfig):

            for k, v in config.tags.items():
                tags[k] = tags[k].union(v)

            config = config.__dict__

        elif isinstance(config, dict) or isinstance(config, Namespace):

            if isinstance(config, Namespace):
                config = vars(config)
            config = copy.deepcopy(config)

            if '_tags' in config:
                for k, v in config['_tags'].items():
                    tags[k] = tags[k].union(v)
                del config['_tags']

        else:
            raise ValueError(f"Invalid hparams type: {type(config)}")

        self._tags = tags

        super().__init__(**config)

    @classmethod
    def default_values(cls):
        return cls(return_defaults=True)

    @classmethod
    def add_argument(cls, name, type, default, help, tags=None):
        if tags is None:
            tags = []
        cls.parameters.append(BeamParam(name, type, default, help, tags=tags))

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
        return (f"{type(self).__name__}:\n\nParameters:\n\n{pformat(self.dict())}\n\n"
                f"Tags:\n\n{pformat(vars(self.tags))}\n\n")

    @property
    def namespace(self):
        return Namespace(**self.__dict__)

    def items(self):
        for k, v in vars(self).items():
            if k.startswith('_'):
                continue
            yield k, v

    @property
    def tags(self):
        return Namespace(**{k: list(v) for k, v in self._tags.items()})

    def keys(self):
        for k in vars(self).keys():
            if k.startswith('_'):
                continue
            yield k

    def values(self):
        for k, v in self.items():
            yield v

    @staticmethod
    def update_parser(parser, defaults=None, parameters=None, source=None):

        if defaults is not None:
            # set defaults
            parser.set_defaults(**{k.replace('-', '_').strip(): v for k, v in defaults.items()})

        if parameters is not None:
            for v in parameters:

                name_to_parse = v.name.replace('_', '-').strip()

                tags = v.tags
                if tags is None:
                    tags = []
                elif isinstance(tags, str):
                    tags = [tags]

                tags = '/'.join(tags)
                if source is not None:
                    tags = f"{source}/{tags}"

                if v.type is bool:
                    boolean_feature(parser, name_to_parse, v.default, v.help)
                elif v.type is list:
                    parser.add_argument(f"--{name_to_parse}", type=v.type, default=v.default, nargs='+', metavar=tags,
                                        help=v.help)
                elif v.type is dict:
                    parser.add_argument(f"--{name_to_parse}", type=json.loads, default=v.default, metavar=tags,
                                        help=v.help)
                else:
                    parser.add_argument(f"--{name_to_parse}", type=v.type, default=v.default, metavar=tags, help=v.help)

        return parser

    def to_path(self, path, ext=None):
        d = copy.deepcopy(self.dict())
        d['_tags'] = self._tags
        beam_path(path).write(d, ext=ext)

    @classmethod
    def from_path(cls, path, ext=None):
        d = beam_path(path).read(ext=ext)
        tags = d.pop('_tags', None)
        return cls(config=d, tags=tags)

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

    def update(self, hparams, tags=None):
        for k, v in hparams.items():
            self.set(k, v, tags=tags)

    def set(self, key, value, tags=None):
        key = key.replace('-', '_').strip()
        setattr(self, key, value)
        if tags is not None:
            if isinstance(tags, str):
                tags = [tags]
            for tag in tags:
                self._tags[tag].add(key)

    def __setattr__(self, key, value):
        key = key.replace('-', '_').strip()
        if key.startswith('_'):
            super().__setattr__(key, value)
        else:
            self._tags['new'].add(key)
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
