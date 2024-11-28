from ..config import BeamConfig, BeamParam
from ..orchestration import ServeClusterConfig

class BeamCICDConfig(BeamConfig):

    parameters = [

        BeamParam('gitlab_url', str, 'https://gitlab.dt.local', 'GitLab URL'),
        BeamParam('gitlab_token', str, 'your_gitlab_token', 'GitLab Token'),
        BeamParam('branch', str, 'main', 'Branch'),
        BeamParam('git_namespace', str, 'dayosupp', 'Namespace'),
        BeamParam('gitlab_project', str, 'dayosupp/yolo', 'GitLab Project'),
        BeamParam('commit_message', str, 'Update CI/CD pipeline configuration', 'Commit Message'),
        BeamParam('image_name', str, 'harbor.dt.local/public/beam:20240801', 'Image Name'),
        BeamParam('registry_user', str, 'admin', 'Registry User'),
        BeamParam('registry_url', str, 'harbor.dt.local', 'Registry URL'),
        BeamParam('registry_name', str, 'Registry User'),
        BeamParam('registry_password', str, 'Registry Password'),
        BeamParam('python_script', str, 'cicd_runner.py', 'Python Script'),
        BeamParam('cmd', str, 'python3', 'Command'),
        BeamParam('python_file', str, 'main.py', 'Python File where the algorithm is defined'),
        BeamParam('python_function', str, 'main', 'Python Function in the Python File which builds the object/algorithm'),
        BeamParam('bash_script', str, 'run_yolo.sh', 'Bash Script'),
        BeamParam('working_dir', str, '/app/', 'Working Directory'),
        BeamParam('ssl_verify', bool, False, 'SSL Verify'),

    ]

class ServeCICDConfig(BeamCICDConfig, ServeClusterConfig):
    pass
