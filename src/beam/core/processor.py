from collections import OrderedDict
import pickle
from argparse import Namespace
import io
from ..path import beam_path, normalize_host
from ..utils import retrieve_name, lazy_property, check_type
from ..config import BeamHparams

try:
    from src.beam.data import BeamData
    has_beam_ds = True
except ImportError:
    has_beam_ds = False


class Processor:

    def __init__(self, *args, name=None, state=None, path=None, hparams=None, override=True, remote=None, **kwargs):

        self._name = name
        self._state = state
        self.remote = remote
        self._path = path

        if len(args) > 0:
            self.hparams = args[0]
        elif hparams is not None:
            self.hparams = hparams
        else:
            if not hasattr(self, 'hparams'):
                self.hparams = BeamHparams(hparams=Namespace())

        for k, v in kwargs.items():
            v_type = check_type(v)
            if v_type.major in ['scalar', 'none']:
                if k not in self.hparams or override:
                    self.hparams[k] = v

    @lazy_property
    def name(self):
        if self._name is None:
            self._name = retrieve_name(self)
        return self._name

    @property
    def state_attributes(self):
        '''
        return of list of class attributes that are used to save the state and the are not part of the
        skeleton of the instance. override this function to add more attributes to the state and avoid pickling a large
        skeleton.
        @return:
        '''

        return []

    @property
    def state(self):
        '''
        return the state of the processor. override this function to add more attributes to the state.
        @return:
        '''
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

    @property
    def path(self):
        return beam_path(self._path)

    @path.setter
    def path(self, value):
        self._path = beam_path(value)

    def __getstate__(self):
        # Create a new state dictionary with only the skeleton attributes without the state attributes
        state = {k: v for k, v in self.__dict__.items() if k not in self.state_attributes}
        return state

    def __setstate__(self, state):
        # Restore the skeleton attributes
        self.__dict__.update(state)

    @classmethod
    def from_remote(cls, hostname, *args, port=None,  **kwargs):

        hostname = normalize_host(hostname, port=port)
        from ..serve.beam_client import BeamClient
        remote = BeamClient(hostname)
        self = cls(*args, remote=remote, **kwargs)

        def detour(self, attr):
            return getattr(self.remote, attr)

        setattr(self, '__getattribute__', detour)

        return self

    def save_state(self, path=None):
        if path is None:
            path = self.path
        path = beam_path(path)

        state = self.state
        if has_beam_ds and isinstance(state, BeamData):
            state.store(path=path)
        else:
            path = beam_path(path)
            path.write(state)

    def state_dict(self):

        state = self.state
        if has_beam_ds and isinstance(self.state, BeamData):
            if not state.is_cached:
                state.cache()
            return state.state_dict()
        else:

            mem_file = io.BytesIO()
            pickle.dump(state, mem_file)

            return {'pickle': mem_file}

    def load_state_dict(self, state_dict):

        if 'pickle' in state_dict and len(state_dict) == 1:
            mem_file = state_dict['pickle']
            mem_file.seek(0)
            self.state = pickle.load(mem_file)

        elif has_beam_ds:
            self.state = BeamData.load_state_dict(state_dict)
        else:
            raise NotImplementedError

    def load_state(self, path=None):
        if path is None:
            path = self.path
        if path is not None:
            path = beam_path(path)

        if path.is_file():
            self.state = path.read()
        elif has_beam_ds:
            state = BeamData.from_path(path=path)
            self.state = state.cache()
        else:
            raise NotImplementedError

    @classmethod
    def from_path(cls, path):
        path = beam_path(path)
        state = path.read()
        return cls(state=state, path=path)

    def get_hparam(self, hparam, default=None, preferred=None, specific=None):
        return self.hparams.get(hparam, default=default, preferred=preferred, specific=specific)


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

