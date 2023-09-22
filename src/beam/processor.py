from .utils import retrieve_name, normalize_host
from collections import OrderedDict
from .path import beam_path
import pickle
import io

try:
    from .data import BeamData
    has_beam_ds = True
except ImportError:
    has_beam_ds = False


class Processor(object):

    def __init__(self, *args, name=None, state=None, path=None, remote=None, **kwargs):

        self.remote = remote

        self._name = name
        self.state = state
        self.path = beam_path(path)

        if len(args) > 0:
            self.hparams = args[0]

        if self.state is None and self.path is not None:
            self.load_state()

    @classmethod
    def from_remote(cls, hostname, *args, port=None,  **kwargs):

        hostname = normalize_host(hostname, port=port)
        from .server import BeamClient
        remote = BeamClient(hostname)
        self = cls(*args, remote=remote, **kwargs)
        return self

    # def __getattribute__(self, name):
    #     try:
    #         remote = super(Processor, self).__getattribute__("remote")
    #         if remote is not None:
    #             return getattr(remote, name)
    #     except:
    #         return super(Processor, self).__getattribute__(name)

    def save_state(self, path=None):
        if path is None:
            path = self.path
        if path is not None:
            path = beam_path(path)

        if has_beam_ds and isinstance(self.state, BeamData):
            self.state.store(path=path)
        else:
            path = beam_path(path)
            path.write(self.state)

    def state_dict(self):
        if has_beam_ds and isinstance(self.state, BeamData):
            if not self.state.is_cached:
                self.state.cache()
            return self.state.state_dict()
        else:

            mem_file = io.BytesIO()
            pickle.dump(self.state, mem_file)

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

    @property
    def name(self):
        if self._name is None:
            self._name = retrieve_name(self)
        return self._name


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

