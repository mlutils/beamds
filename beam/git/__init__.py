if len([]):

    from .cicd import BeamCICD
    # from .gitlab import BeamGitlab
    # from .github import BeamGithub
    from .core import BeamGit
    from .config import BeamCICDConfig

__all__ = ['BeamCICD', 'BeamGit', 'BeamCICDConfig']

def __getattr__(name):
    if name == 'BeamCICD':
        from .cicd import BeamCICD
        return BeamCICD
    elif name == 'BeamGit':
        from .core import BeamGit
        return BeamGit
    elif name == 'BeamCICDConfig':
        from .config import BeamCICDConfig
        return BeamCICDConfig
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")

