from contextlib import contextmanager
from uuid import uuid4 as uuid

from .models import BeamPath
from .resource import beam_path


@contextmanager
def local_copy(path, tmp_path='/tmp', as_beam_path=False, override=False, disable=None):

    path = beam_path(path)

    if disable is None:
        disable = path.scheme == 'file'

    if disable:
        if as_beam_path:
            yield path
        else:
            yield str(path) if path else None
        return

    path = beam_path(path)
    tmp_dir = beam_path(tmp_path).joinpath(uuid())
    tmp_dir.mkdir(exist_ok=True, parents=True)
    tmp_path = tmp_dir.joinpath(path.name)

    exists = path.exists() and (path.is_file() or len(list(path)) > 0)

    # assert not exists or override, f'Path {path} already exists, set override=True to overwrite it.'

    if exists:
        path.copy(tmp_path)

    try:
        yield tmp_path if as_beam_path else str(tmp_path)
    finally:
        if override or not exists:
            tmp_path.copy(path)
            tmp_dir.rmtree()


class FileSystem:
    def __init__(self, path):
        self.path = beam_path(path)

    def exists(self):
        return self.path.exists()

    def is_file(self):
        return self.path.is_file()

    def is_dir(self):
        return self.path.is_dir()

    def joinpath(self, path):
         self.path = self.path.joinpath(path)

    def rmtree(self):
        self.path.rmtree()

    def mkdir(self, exist_ok=True, parents=True):
        self.path.mkdir(exist_ok=exist_ok, parents=parents)

    def copy(self, path):
        path = beam_path(path)
        path.parent.mkdir(exist_ok=True, parents=True)
        self.path.copy(path)

    def open(self, mode='r'):
        return self.path.open(mode=mode)

    def root(self):
        self.path = self.path.root()

    def __str__(self):
        return str(self.path)

    def __repr__(self):
        return repr(self.path)

    def __iter__(self):
        for p in self.path:
            yield p.name

    def iterdir(self):
        return self.__iter__()


@contextmanager
def temp_local_file(content, tmp_path='/tmp', name=None, ext=None, binary=True, as_beam_path=True):
    tmp_path = BeamPath(tmp_path).joinpath(uuid())
    tmp_path.mkdir(exist_ok=True, parents=True)
    if name is None:
        name = uuid()
    if ext is not None:
        name = f"{name}{ext}"
    tmp_path = tmp_path.joinpath(name)
    try:
        if binary:
            tmp_path.write_bytes(content)
        else:
            tmp_path.write_text(content)
        yield tmp_path if as_beam_path else str(tmp_path)
    finally:
        tmp_path.unlink()
        tmp_path.parent.rmdir()
