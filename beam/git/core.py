import git
import tempfile

from ..base import BeamBase, tmp_paths
from ..path import beam_path


class BeamGit(BeamBase):

    def __init__(self, repo: str|None = None, branch: str|None = None, path: str|None = None, api_key=None, *args, **kwargs):
        super().__init__(*args, repo=repo, branch=branch, path=path, api_key=api_key, **kwargs)
        self.repo_name = self.get_hparam('repo')
        self.branch = self.get_hparam('branch')
        self.path = self.get_hparam('path')
        self.api_key = self.get_hparam('api_key')
        self._is_tmp_path = False
        self._repo = None
        if self.path is None:
            path = beam_path(tempfile.mkdtemp(dir=tmp_paths.code_repos, prefix=self.project_name))
            self.path = beam_path(path).joinpath(self.project_name)
            self._is_tmp_path = True

    @property
    def project_name(self):
        return self.repo.split('/')[-1].replace('.git', '')

    def clone(self):
        return git.Repo.clone_from(self.repo_name, self.path.parent.str, branch=self.branch)

    @staticmethod
    def is_git_repo(path: str):
        try:
            git.Repo(path)
            return True
        except git.InvalidGitRepositoryError:
            return False

    @property
    def repo(self):
        if self._repo is None:
            if not self.is_git_repo(self.path):
                self._repo = self.clone()
            else:
                self._repo = git.Repo(self.path)
        return self._repo

    def checkout(self, branch: str):
        repo = git.Repo(self.path)
        repo.git.checkout(branch)

    def commit(self, message: str):
        self.repo.git.add(update=True)
        self.repo.git.commit(message=message)

    def push(self):
        self.repo.git.push()

    def pull(self):
        self.repo.git.pull()

    def __enter__(self):
        self.clone()
        return self.path

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._is_tmp_path:
            self.path.parent.rmtree()

