from ..base import BeamBase
import git
import tempfile


class BeamGit(BeamBase):

    def __init__(self, repo: str|None = None, branch: str|None = None, path: str|None = None, *args, **kwargs):
        super().__init__(*args, repo=repo, branch=branch, path=path, **kwargs)
        self.repo = self.get_hparam('repo')
        self.branch = self.get_hparam('branch')
        self.path = self.get_hparam('path')
        if self.path is None:
            self.path = tempfile.mkdtemp(dir='')

    def clone(self):
        git.Repo.clone_from(self.repo, self.path, branch=self.branch)

