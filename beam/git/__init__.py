if len([]):

    from .cicd_client import BeamCICDClient
    # from .gitlab import BeamGitlab
    # from .github import BeamGithub
    from .core import BeamGit
    from .config import BeamCICDConfig, ServeCICDConfig
    from .git_resource import deploy_cicd

__all__ = ['BeamCICDClient', 'BeamGit', 'deploy_cicd', 'BeamCICDConfig', 'ServeCICDConfig']

def __getattr__(name):
    if name == 'BeamCICDClient':
        from .cicd_client import BeamCICDClient
        return BeamCICDClient
    elif name == 'BeamGit':
        from .core import BeamGit
        return BeamGit
    elif name == 'BeamCICDConfig':
        from .config import BeamCICDConfig
        return BeamCICDConfig
    elif name == 'ServeCICDConfig':
        from .config import ServeCICDConfig
        return ServeCICDConfig
    elif name == 'deploy_cicd':
        from .git_resource import deploy_cicd
        return deploy_cicd
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")

