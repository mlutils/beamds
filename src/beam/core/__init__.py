from ..path import beam_path
from ..utils import retrieve_name, get_cached_properties


class MetaBeamInit(type):
    def __call__(cls, *args, _store_init_path=None, **kwargs):
        init_args = {'args': args, 'kwargs': kwargs}
        if _store_init_path:
            cls._pre_init(_store_init_path, init_args)
        instance = super().__call__(*args, **kwargs)
        instance._init_args = init_args
        instance._init_is_done = True
        return instance

    def _pre_init(cls, store_init_path, init_args):
        # Process or store arguments
        store_init_path = beam_path(store_init_path)
        store_init_path.write(init_args, ext='.pkl')


class BeamBase(metaclass=MetaBeamInit):

    def __init__(self, *args, name=None, **kwargs):

        self._init_is_done = False
        self._name = name

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

    def beam_class(self):
        return [c.__name__ for c in self.__class__.mro()]