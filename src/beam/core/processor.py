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

    def __init__(self, *args, name=None, hparams=None, override=True, remote=None, **kwargs):

        self._name = name
        self.remote = remote
        self._lazy_cache = {}

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

    def get_hparam(self, hparam, default=None, preferred=None, specific=None):
        return self.hparams.get(hparam, default=default, preferred=preferred, specific=specific)

    @classmethod
    def from_path(cls, path):
        path = beam_path(path)
        state = path.read()
        hparams = BeamHparams(hparams=state['hparams'])
        alg = cls(hparams)
        alg.load_state(state)
        return alg

    def state_dict(self):
        # The state must contain a key 'hparams' with the hparams of the instance
        raise NotImplementedError

    def load_state_dict(self, state_dict):
        raise NotImplementedError

    def save_state(self, path, ext=None):

        state = self.state_dict()

        path = beam_path(path)
        if has_beam_ds and isinstance(state, BeamData):
            state.store(path=path, file_type=ext)
        elif has_beam_ds and (not path.suffix) and ext is None:
            state = BeamData(data=state, path=path)
            state.store()
        else:
            path.write(state, ext=ext)

    def load_state(self, path):

        path = beam_path(path)

        if path.is_file():
            state = path.read()
        elif has_beam_ds:
            state = BeamData.from_path(path=path)
            state = state.cache()
        else:
            raise NotImplementedError

        self.load_state_dict(state)

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

