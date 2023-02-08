from pathlib import PurePath


class BeamPath(PurePath):

    def __init__(self, *pathsegments):
        super(BeamPath, self).__init__(*pathsegments)

    def samefile(self, other):
        raise NotImplementedError

    def is_file(self):
        raise NotImplementedError

    def is_dir(self):
        raise NotImplementedError

    def joinpath(self, *other):
        raise NotImplementedError

    def mkdir(self, *args, **kwargs):
        raise NotImplementedError

    def exists(self):
        raise NotImplementedError

    def glob(self, *args, **kwargs):
        raise NotImplementedError


class HDFSPath(BeamPath):

    def __init__(self, *pathsegments, client=None):
        super(HDFSPath, self).__init__(*pathsegments)
        self.client = client

    def iterdir(self):
        files = self.client.list(str(self))
        for f in files:
            yield HDFSPath(self.joinpath(f), client=self.client)

    def samefile(self, other):
        raise NotImplementedError

    def is_file(self):
        raise NotImplementedError

    def is_dir(self):
        raise NotImplementedError

    def joinpath(self, *other):
        raise NotImplementedError

    def mkdir(self, *args, **kwargs):
        raise NotImplementedError

    def exists(self):
        raise NotImplementedError

    def glob(self, *args, **kwargs):
        raise NotImplementedError
