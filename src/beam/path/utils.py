from contextlib import contextmanager
from uuid import uuid4 as uuid
from .resource import beam_path

@contextmanager
def local_copy(path, tmp_path='/tmp', as_beam_path=True):
    path = beam_path(path)
    tmp_dir = beam_path(tmp_path).joinpath(uuid())
    tmp_dir.mkdir(exist_ok=True, parents=True)
    tmp_path = tmp_dir.joinpath(path.name)
    path.copy(tmp_path)
    try:
        yield tmp_path if as_beam_path else str(tmp_path)
    finally:
        tmp_dir.rmtree()
