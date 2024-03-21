import json
from collections import OrderedDict
import inspect
import pickle
from argparse import Namespace
import io
from ..path import beam_path, normalize_host
from ..utils import retrieve_name, lazy_property, check_type
from ..config import BeamConfig


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
        self._lazy_cache = {}

    def getattr(self, attr):
        raise AttributeError(f"Attribute {attr} not found")

    def __getattr__(self, item):
        if (item.startswith('_') or item == '_init_is_done' or not hasattr(self, '_init_is_done')
                or not self._init_is_done):
            return object.__getattribute__(self, item)
        return self.getattr(item)

    def clear_cache(self, *args):
        if len(args) == 0:
            self._lazy_cache = {}
        else:
            for k in args:
                if k in self._lazy_cache:
                    del self._lazy_cache[k]

    def in_cache(self, attr):
        return attr in self._lazy_cache

    @property
    def name(self):
        if self._name is None and hasattr(self, '_init_is_done') and self._init_is_done:
            self._name = retrieve_name(self)
        return self._name

    def beam_class(self):
        return self.__class__.__name__


class Processor(BeamBase):

    skeleton_file = 'skeleton'
    state_file = 'state'

    def __init__(self, *args, name=None, hparams=None, override=True, remote=None, llm=None, **kwargs):

        super().__init__(name=name)
        self.remote = remote

        if len(args) > 0:
            self.hparams = args[0]
        elif hparams is not None:
            self.hparams = hparams
        else:
            if not hasattr(self, 'hparams'):
                self.hparams = BeamConfig(config=Namespace())

        for k, v in kwargs.items():
            v_type = check_type(v)
            if v_type.major in ['scalar', 'none']:
                if k not in self.hparams or override:
                    self.hparams[k] = v

        self._llm = self.get_hparam('llm', llm)

    @lazy_property
    def llm(self):
        if type(self._llm) is str:
            from ..resource import resource
            self._llm = resource(self._llm)
        return self._llm

    @property
    def exclude_pickle_attributes(self):
        '''
        return of list of class attributes that are used to save the state and are not part of the
        skeleton of the instance. override this function to add more attributes to the state and avoid pickling a large
        skeleton.
        @return:
        '''
        keys = self.state_dict().keys()
        keys = [k for k in keys if k not in ['hparams']]

        return keys

    def __getstate__(self):
        # Create a new state dictionary with only the skeleton attributes without the state attributes
        state = {k: v for k, v in self.__dict__.items() if k not in self.exclude_pickle_attributes}
        return state

    def __setstate__(self, state):
        # Restore the skeleton attributes
        self.__dict__.update(state)

    @classmethod
    def from_remote(cls, hostname, *args, port=None,  **kwargs):

        hostname = normalize_host(hostname, port=port)
        from ..serve.client import BeamClient
        remote = BeamClient(hostname)
        self = cls(*args, remote=remote, **kwargs)

        def detour(self, attr):
            return getattr(self.remote, attr)

        setattr(self, '__getattribute__', detour)

        return self

    @classmethod
    def from_arguments(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    @classmethod
    def from_path(cls, path):
        path = beam_path(path)

        skeleton_file = list(path.glob(f"{Processor.skeleton_file}*"))[0]
        state_file = list(path.glob(f"{Processor.state_file}*"))[0]

        obj = skeleton_file.read()
        obj.load_state(state_file)

        return obj

    @classmethod
    def from_state_path(cls, path):
        path = beam_path(path)
        state = path.read()
        kwargs = dict()
        args = tuple()
        if 'aux' in state:
            if 'kwargs' in state['aux']:
                kwargs = state['aux']['kwargs']
            if 'args' in state['aux']:
                args = state['aux']['args']
        hparams = BeamConfig(config=state['hparams'])
        alg = cls(hparams, *args, **kwargs)
        alg.load_state(state)
        return alg

    @classmethod
    def from_nlp(cls, query, llm=None, ask_kwargs=None, **kwargs):
        from ..resource import resource
        from ..logger import beam_logger as logger

        llm = resource(llm)

        def is_class_method(member):
            # First, ensure that member is a method bound to a class
            if inspect.ismethod(member) and inspect.isclass(member.__self__):
                # Now that we've confirmed member is a method, check the name conditions
                if not member.__name__.startswith('_') and member.__name__ != 'from_nlp':
                    return True
            return False

        classmethods = [name for name, member in inspect.getmembers(cls, predicate=is_class_method)]

        example_output = {'method': 'method_name'}
        prompt = (f"Choose the suitable classmethod that should be used to build a class instance according to the "
                  f"following query:\n"
                  f"Query: {query}\n"
                  f"Class: {cls.__name__}\n"
                  f"Methods: {classmethods}\n"
                  f"Return your answer as a JSON object of the following form:\n"
                  f"{json.dumps(example_output)}\n"
                  f"Your answer:\n\n")

        ask_kwargs = ask_kwargs or {}
        response = llm.ask(prompt, **ask_kwargs).json

        constructor_name = response['method']

        if constructor_name not in classmethods:
            raise ValueError(f"Constructor {constructor_name} not found in the list of class constructors")

        constructor = getattr(cls, constructor_name)
        logger.info(f"Using classmethod {constructor_name} to build the class instance")

        constructor_sourcecode = inspect.getsource(constructor)
        init_sourcecode = inspect.getsource(cls.__init__)

        json_output_example = {"args": ['arg1', 'arg2'], "kwargs": {'kwarg1': 'value1', 'kwarg2': 'value2'}}
        prompt = (f"Build a suitable dictionary of arguments and keyword arguments to build a class instance according "
                  f"to the following query:\n"
                  f"Query: {query}\n"
                  f"with the classmethod: {constructor_name} (of class {cls.__name__}) with source-code:\n"
                  f"{constructor_sourcecode}\n"
                  f"and the class __init__ method source-code:\n"
                  f"{init_sourcecode}\n"
                  f"Return your answer as a JSON object of the following form:\n"
                  f"{json_output_example}\n"
                  f"Your answer:\n\n")

        d = llm.ask(prompt, **ask_kwargs).json
        args = d.get('args', [])
        kwargs = d.get('kwargs', {})

        logger.info(f"Using args: {args} and kwargs: {kwargs} to build the class instance")

        return constructor(*args, **kwargs)

    def get_hparam(self, hparam, default=None, preferred=None, specific=None):
        return self.hparams.get(hparam, default=default, preferred=preferred, specific=specific)

    def set_hparam(self, hparam, value, tags=None):
        self.hparams.set(hparam, value, tags=tags)

    def update_hparams(self, hparams, tags=None):
        self.hparams.update(hparams, tags=tags)

    def to_bundle(self, path):
        from ..auto import AutoBeam
        AutoBeam.to_bundle(self, path)

    def state_dict(self):
        # The state must contain a key 'hparams' with the hparams of the instance
        return {'hparams': self.hparams}

    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            setattr(self, k, v)

    def save_state(self, path, ext=None):

        state = self.state_dict()
        path = beam_path(path)

        try:
            from ..data import BeamData
            has_beam_ds = True
        except ImportError:
            has_beam_ds = False

        if has_beam_ds and isinstance(state, BeamData):
            state.store(path=path, file_type=ext)
        elif has_beam_ds and (not path.suffix) and ext is None:
            state = BeamData(data=state, path=path)
            state.store()
        else:
            path.write(state, ext=ext)

    def to_path(self, path, skeleton_ext=None, state_ext=None):
        path = beam_path(path)
        path.clean()
        path.mkdir()
        skeleton_ext = skeleton_ext or '.pkl'
        state_ext = state_ext or ''
        skeleton_file = f"{Processor.skeleton_file}{skeleton_ext}"
        state_file = f"{Processor.state_file}{state_ext}"

        path.joinpath(skeleton_file).write(self)
        self.save_state(path.joinpath(state_file))

    def load_state(self, path):

        path = beam_path(path)

        try:
            from ..data import BeamData
            has_beam_ds = True
        except ImportError:
            has_beam_ds = False

        if path.is_file():
            state = path.read()
        elif has_beam_ds:
            state = BeamData.from_path(path=path)
            state = state.cache()
        else:
            raise NotImplementedError

        self.load_state_dict(state)

    def nlp(self, query, llm=None, ask_kwargs=None, **kwargs):

        from ..logger import beam_logger as logger

        if llm is None:
            llm = self.llm
        elif type(llm) is str:
            from ..resource import resource
            llm = resource(llm)

        ask_kwargs = ask_kwargs or {}

        method_list = inspect.getmembers(self, predicate=inspect.isroutine)
        method_list = [m for m in method_list if not m[0].startswith('_')]
        json_output_example = json.dumps({'method': 'method_name'})
        class_doc = inspect.getdoc(self)
        class_doc = f"{class_doc}\n" if class_doc else ""

        prompt = (f"Choose the suitable method that should be used to answer the following query:\n"
                  f"Query: {query}\n"
                  f"Class: {self.__class__.__name__}\n"
                  f"{class_doc}"
                  f"Attributes: {method_list}\n"
                  f"Return your answer as a JSON object of the following form:\n"
                  f"{json_output_example}\n"
                  f"Your answer:\n\n")

        response = llm.ask(prompt, **ask_kwargs).json
        method_name = response['method']

        if method_name not in [m[0] for m in method_list]:
            raise ValueError(f"Method {method_name} not found in the list of methods")

        logger.info(f"Using method {method_name} to answer the query")

        method = getattr(self, method_name)
        sourcecode = inspect.getsource(method)

        json_output_example = {"args": ['arg1', 'arg2'], "kwargs": {'kwarg1': 'value1', 'kwarg2': 'value2'}}

        prompt = (f"Build a suitable dictionary of arguments and keyword arguments to answer the following query:\n"
                  f"Query: {query}\n"
                  f"with the class method: {method_name} (of class {self.__class__.__name__}) with source-code:\n"
                  f"{sourcecode}\n"
                  f"Return your answer as a JSON object of the following form:\n"
                  f"{json_output_example}\n"
                  f"Your answer:\n\n")

        d = llm.ask(prompt, **ask_kwargs).json

        args = d.get('args', [])
        kwargs = d.get('kwargs', {})

        logger.info(f"Using args: {args} and kwargs: {kwargs} to answer the query")

        return method(*args, **kwargs)


class Pipeline(Processor):

    def __init__(self, hparams, *ts, track_steps=False, name=None, state=None, path=None, **kwts):

        super().__init__(hparams, name=name, state=state, path=path)
        self.track_steps = track_steps
        self.steps = {}

        self.transformers = OrderedDict()
        for i, t in enumerate(ts):
            self.transformers[i] = t

        for k, t in kwts.items():
            self.transformers[k] = t

    def transform(self, x, **kwargs):

        self.steps = []

        for i, t in self.transformers.items():

            kwargs_i = kwargs[i] if i in kwargs.keys() else {}
            x = t.transform(x, **kwargs_i)

            if self.track_steps:
                self.steps[i] = x

        return x

