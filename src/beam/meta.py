from .utils import retrieve_name


class MetaBeamInit(type):
    def __call__(cls, *args, _store_init_path=None, _save_init_args=False, **kwargs):

        init_args = None
        if _store_init_path or _save_init_args:
            init_args = {'args': args, 'kwargs': kwargs}
        if _store_init_path:
            cls._pre_init(_store_init_path, init_args)
        instance = super().__call__(*args, **kwargs)
        instance._init_args = init_args
        instance._init_is_done = True
        return instance

    def _pre_init(cls, store_init_path, init_args):
        # Process or store arguments
        from .path import beam_path
        store_init_path = beam_path(store_init_path)
        store_init_path.write(init_args, ext='.pkl')


class BeamName:

    def __init__(self, name=None):
        self._name = name

    @property
    def name(self):
        if self._name is None:
            self._name = retrieve_name(self)
        return self._name

    def set_name(self, name):
        self._name = name

    @property
    def beam_class_name(self):
        return [c.__name__ for c in self.__class__.mro()]
