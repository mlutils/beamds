import inspect
from argparse import Namespace
from functools import cached_property

from ..type import check_type
from ..meta import MetaBeamInit, BeamName

from ..utils import get_cached_properties, is_notebook
from ..config import BeamConfig


class BeamBase(BeamName, metaclass=MetaBeamInit):

    def __init__(self, *args, name=None, hparams=None, _init_args=None, _config_scheme=None, **kwargs):

        super().__init__(name=name)
        self._init_is_done = False
        _init_args = _init_args or {}

        config_scheme = _config_scheme or BeamConfig
        if len(args) > 0 and (isinstance(args[0], BeamConfig) or isinstance(args[0], dict)):
            self.hparams = config_scheme(args[0])
        elif hparams is not None:
            self.hparams = config_scheme(hparams)
        else:
            if not hasattr(self, 'hparams'):
                self.hparams = config_scheme()

        for k, v in kwargs.items():
            if not k.startswith('_'):
                v_type = check_type(v)
                if v_type.major in ['scalar', 'none']:
                    if k in _init_args.get('kwargs', {}) or k not in self.hparams or self._default_value(k) != v:
                        self.hparams[k] = v

    @cached_property
    def _signatures(self):
        sigs = []
        for c in self.__class__.mro():
            sigs.append(inspect.signature(c.__init__))
        return sigs

    def _default_value(self, key):
        default = None
        for s in self._signatures:
            if key in s.parameters:
                default = s.parameters[key].default
                break
        return default

    def getattr(self, attr):
        raise AttributeError(f"Attribute {attr} not found.\n"
                             f"For cached_property attributes, it is possible to reach here if an AttributeError is "
                             f"raised in the getter function.")

    def __getattr__(self, item):
        if item.startswith('_') or item == '_init_is_done' or not self.is_initialized:
            return object.__getattribute__(self, item)
        return self.getattr(item)

    @property
    def is_initialized(self):
        return hasattr(self, '_init_is_done') and self._init_is_done

    def clear_cache(self, *args):
        if len(args) == 0:
            args = get_cached_properties(self)
        for k in args:
            try:
                delattr(self, k)
            except AttributeError:
                pass

    def in_cache(self, attr):
        return hasattr(self, attr)

    @cached_property
    def is_notebook(self):
        return is_notebook()
