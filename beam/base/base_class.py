import copy
import inspect
from argparse import Namespace

from ..type import check_type
from ..meta import MetaBeamInit, BeamName
from ..type.utils import is_beam_config, Types

from ..utils import get_cached_properties, is_notebook, cached_property
from ..config import BeamConfig


class BeamBase(BeamName, metaclass=MetaBeamInit):

    def __init__(self, *args, name=None, hparams=None, _init_args=None, _config_scheme=None, **kwargs):

        # directly call the parent class __init__ method
        super().__init__(name=name, **kwargs)

        self._init_is_done = False
        _init_args = _init_args or {}

        config_scheme = _config_scheme or BeamConfig
        if hparams is not None:
            if is_beam_config(hparams):
                self.hparams = copy.deepcopy(hparams)
            else:
                self.hparams = config_scheme(hparams, load_script_arguments=False, load_config_files=False)
        elif len(args) > 0 and is_beam_config(args[0]):
            self.hparams = copy.deepcopy(args[0])
        elif len(args) > 0 and isinstance(args[0], (list, tuple, set, dict, Namespace)):
            self.hparams = config_scheme(args[0], load_script_arguments=False, load_config_files=False)
        else:
            if not hasattr(self, 'hparams'):
                self.hparams = config_scheme(load_script_arguments=False, load_config_files=False)

        _init_kwargs = _init_args.get('kwargs', {})
        for k, v in kwargs.items():
            if not k.startswith('_'):
                v_type = check_type(v)
                if v_type.major in [Types.scalar, Types.none]:
                    if k in _init_kwargs or k not in self.hparams or self._default_value(k) != v:
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

    def hasattr(self, attr):
        return attr in self.__dict__

    def getattr(self, attr):
        # Capture the full traceback
        # tb = ''.join(traceback.format_stack())
        # raise AttributeError(f"Attribute {attr} not found.\n"
        #                      f"For cached_property attributes, it is possible to reach here if an AttributeError is "
        #                      f"raised in the getter function.\n"
        #                      f"Traceback:\n{tb}")

        raise AttributeError(f"Attribute {attr} not found.\n"
                             f"For cached_property attributes, it is possible to reach here if an AttributeError is "
                             f"raised in the getter function (to debug, set traceback in method: {attr}).")

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

    def get_hparam(self, hparam, default=None, specific=None):
        return self.hparams.get(hparam, default=default, specific=specific)

    def set_hparam(self, hparam, value, tags=None):
        self.hparams.set(hparam, value, tags=tags)

    def update_hparams(self, hparams, tags=None):
        self.hparams.update(hparams, tags=tags)
