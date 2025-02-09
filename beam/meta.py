from .utils import retrieve_name


class MetaBeamInit(type):
    def __call__(cls, *args, _store_init_path=None, _save_init_args=True, **kwargs):

        init_args = {'args': args, 'kwargs': kwargs}
        if _store_init_path:
            cls._pre_init(_store_init_path, init_args)
        if '_init_args' not in kwargs:
            kwargs['_init_args'] = init_args

        instance = super().__call__(*args, **kwargs)
        instance._init_args = init_args if _save_init_args else None
        instance._init_is_done = True

        return instance

    def _pre_init(cls, store_init_path, init_args):
        # Process or store arguments
        from .path import beam_path
        store_init_path = beam_path(store_init_path)
        store_init_path.write(init_args, ext='.pkl')


class BeamName:

    def __init__(self, name=None, dynamic_name=True, **kwargs):

        # Get the next class in the MRO
        mro = type(self).mro()
        base = mro[mro.index(BeamName) + 1]
        if base == object:
            super().__init__()
        else:
            super().__init__(**kwargs)

        self._name = name
        self._dynamic_name = dynamic_name

    @property
    def name(self):
        if self._name is None and self._dynamic_name:
            self._name = retrieve_name(self)
        return self._name

    def set_name(self, name):
        self._name = name

    @property
    def beam_class_name(self):
        return [c.__name__ for c in self.__class__.mro()]

