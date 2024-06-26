import json
from collections import OrderedDict
import inspect
from contextlib import contextmanager
from functools import cached_property
from typing import List, Union

from ..path import beam_path, normalize_host
from ..config import BeamConfig
from ..base import BeamBase
from ..type.utils import is_beam_processor
from ..data import BeamData


class Processor(BeamBase):

    skeleton_file = '_skeleton'
    init_args_file = '_init_args'

    def __init__(self, *args, name=None, llm=None, **kwargs):

        super().__init__(*args, name=name, llm=llm, **kwargs)
        self._llm = self.get_hparam('llm', llm)
        self._beam_pickle = False

    @contextmanager
    def beam_pickle(self, on=True):
        prev = self._beam_pickle
        self._beam_pickle = on
        yield
        self._beam_pickle = prev

    def in_beam_pickle(self):
        return self._beam_pickle

    @cached_property
    def llm(self):
        if type(self._llm) is str:
            from ..resource import resource
            self._llm = resource(self._llm)
        return self._llm

    @classmethod
    @property
    def special_state_attributes(cls):
        '''
        return of list of special class attributes that are stored individually in the state and not as part of the
        skeleton of the instance (i.e. a pickle object).
        override this function to add more attributes to the state and avoid pickling a large skeleton.
        @return:
        '''
        return ['hparams']

    @classmethod
    @property
    def excluded_attributes(cls):
        '''
        return of list of class attributes should not be saved in the state. override this function to exclude some
        attributes from the state.
        @return:
        '''
        return ['_init_args', '_skeleton']

    def __getstate__(self):
        # Create a new state dictionary with only the skeleton attributes without the state attributes
        # this is a mislead name, as __getstate__ is used to get the skeleton of the instance and not the state
        if self.in_beam_pickle():
            with self.beam_pickle(on=False):
                excliuded_attributes = [*self.excluded_attributes, *self.special_state_attributes]
                state = {k: v for k, v in self.__dict__.items() if k not in excliuded_attributes}
                state = state.copy()
        else:
            state = self.__dict__.copy()
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
    def from_path(cls, path, skeleton: Union[bool, str] = True, init_args: Union[bool, str] = True,
                  load_state_kwargs=None, exclude: List = None,  **kwargs):

        load_state_kwargs = load_state_kwargs or {}
        exclude = exclude or []
        exclude = exclude + cls.excluded_attributes
        path = beam_path(path)

        if skeleton:
            if skeleton is True:
                skeleton = Processor.skeleton_file
            obj = BeamData.read(path.joinpath(skeleton), **kwargs)
            if obj is not None:
                obj.load_state(path, skeleton=False, **load_state_kwargs)
                return obj

            # if path.joinpath(skeleton).exists():
            #     obj = path.joinpath(skeleton).read()
            #     obj.load_state(path, skeleton=False, **load_state_kwargs)
            #     return obj

        state = cls.load_state(path, exclude=exclude, **kwargs)

        obj = None
        if init_args:
            if init_args is True:
                init_args = Processor.init_args_file
            d = BeamData.read(path.joinpath(init_args), **kwargs)
            if init_args is not None:
                init_args = d['args']
                init_kwargs = d['kwargs']
                obj = cls(*init_args, **init_kwargs)

            # if path.joinpath(init_args).exists():
            #     d = path.joinpath(init_args).read()
            #     init_args = d['args']
            #     init_kwargs = d['kwargs']
            #     obj = cls(*init_args, **init_kwargs)

        if obj is None:
            init_args = []
            init_kwargs = {}
            hparams = BeamConfig(config=state['hparams'] if 'hparams' in state else {})
            init_kwargs['hparams'] = hparams
            init_kwargs.update(kwargs)
            obj = cls(*init_args, **init_kwargs)

        obj.load_state(state=state, **load_state_kwargs)

        return obj

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

    def hasattr(self, attr):
        return attr in self.__dict__

    def load_state_dict(self, path, ext=None, exclude: List = None, hparams=True, exclude_hparams=None, **kwargs):

        exclude = exclude or []
        exclude = [*exclude, *self.excluded_attributes]
        path = beam_path(path)
        ext = ext or path.suffix

        state = {}
        if ext and ext != '.bmp':
            state = path.read(ext=ext, **kwargs)
        else:
            if path.is_dir() and path.suffix not in ['.bmd']:
                for p in path.iterdir():
                    k = p.stem
                    if k not in exclude:
                        if self.hasattr(k) and is_beam_processor(getattr(self, k)):
                            v = getattr(self, k)
                            v.load_state(p, **kwargs)
                            state[k] = v
                        else:
                            state[k] = BeamData.read(p, **kwargs)
            else:
                state = BeamData.read(path, **kwargs)

        if exclude:
            state = {k: v for k, v in state.items() if k not in exclude}

        for k, v in state.items():
            if k == 'hparams' and hasattr(self, 'hparams'):
                if hparams:
                    exclude_hparams = exclude_hparams or []
                    self.hparams.update(v, exclude=exclude_hparams)
            else:
                setattr(self, k, v)

    def save_state_dict(self, state, path, ext=None, exclude: List = None, override=False, **kwargs):

        path = beam_path(path)
        ext = ext or path.suffix
        exclude = exclude or []
        exclude = [*exclude, *self.excluded_attributes]

        state = {k: v for k, v in state.items() if k not in exclude}

        if ext and ext != '.bmp':
            path.write(state, ext=ext, **kwargs)
        else:
            BeamData.write_tree(state, path, override=override, split=False, archive_size=0)

    def save_state(self, path, ext=None, exclude: List = None, skeleton: Union[bool, str] = True,
                   init_args: Union[bool, str] = False, override=False, **kwargs):
        state = {}
        exclude = exclude or []
        exclude = [*exclude, *self.excluded_attributes]

        for n in self.special_state_attributes:
            # save only cached_properties that are already computed
            if n not in self.excluded_attributes and self.hasattr(n):
                state[n] = getattr(self, n)

        self.save_state_dict(state, path, ext=ext, exclude=exclude, override=override, **kwargs)
        path = self.base_dir(path, ext=ext)

        if skeleton:
            if skeleton is True:
                skeleton = Processor.skeleton_file
            with self.beam_pickle():
                BeamData.write_object(self, path.joinpath(skeleton),
                                      override=override, split=False, archive_size=0)

                # if override or not path.joinpath(skeleton).exists():
                #     path.joinpath(skeleton).write(self)
                # else:
                #     from ..logger import beam_logger as logger
                #     logger.warning(f"Skeleton file: {path.joinpath(skeleton)} already exists, skipping")

        if init_args:
            if init_args is True:
                init_args = Processor.init_args_file

            BeamData.write_object(self._init_args, path.joinpath(init_args),
                                      override=override, split=False, archive_size=0)

            # if override or not path.joinpath(init_args).exists():
            #     path.joinpath(init_args).write(self._init_args)
            # else:
            #     from ..logger import beam_logger as logger
            #     logger.warning(f"Init_args file: {path.joinpath(init_args)} already exists, skipping")

    @staticmethod
    def base_dir(path, ext=None):
        path = beam_path(path)
        ext = ext or path.suffix
        if ext and ext != '.bmp':
            # to load the skeleton and the init_args in the same directory as the state file
            path = path.parent.joinpath(f".{path.stem}")

        return path

    def load_state(self, path=None, state=None, ext=None, exclude: List = None, skeleton: Union[bool,str] = True,
                   **kwargs):


        assert path or state, 'Either path or state must be provided'

        exclude = exclude or []
        exclude = [*exclude, *self.excluded_attributes]
        path = beam_path(path)
        if state is None:
            self.load_state_dict(path=path, ext=ext, exclude=exclude, **kwargs)
            path = self.base_dir(path, ext=ext)

        if skeleton:
            if skeleton is True:
                skeleton = Processor.skeleton_file

            skeleton = BeamData.read(path.joinpath(skeleton), **kwargs)
            self.__dict__.update(skeleton.__dict__)

            # if path.joinpath(skeleton).exists():
            #     skeleton = path.joinpath(skeleton).read()
            #     self.__dict__.update(skeleton.__dict__)

    def to_path(self, path, **kwargs):
        self.save_state(path, **kwargs)

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

