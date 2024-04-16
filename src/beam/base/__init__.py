from argparse import Namespace

from ..type import check_type
from ..meta import MetaBeamInit
from ..utils import retrieve_name, get_cached_properties
from ..config import BeamConfig


class BeamBase(metaclass=MetaBeamInit):

    def __init__(self, *args, name=None, override=True, hparams=None, **kwargs):

        self._init_is_done = False
        self._name = name

        if len(args) > 0 and (isinstance(args[0], BeamConfig) or isinstance(args[0], dict)):
            self.hparams = BeamConfig(args[0])
        elif hparams is not None:
            self.hparams = BeamConfig(hparams)
        else:
            if not hasattr(self, 'hparams'):
                self.hparams = BeamConfig(config=Namespace())

        for k, v in kwargs.items():
            v_type = check_type(v)
            if v_type.major in ['scalar', 'none']:
                if k not in self.hparams or override:
                    self.hparams[k] = v

    def getattr(self, attr):
        raise AttributeError(f"Attribute {attr} not found")

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
            if hasattr(self, k):
                delattr(self, k)

    def in_cache(self, attr):
        return hasattr(self, attr)

    @property
    def name(self):
        if self._name is None and self.is_initialized:
            self._name = retrieve_name(self)
        return self._name

    @property
    def beam_class(self):
        return [c.__name__ for c in self.__class__.mro()]