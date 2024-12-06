from ..config import BeamConfig, BeamParam
from ..orchestration import ServeClusterConfig

class BeamCICDConfig(BeamConfig):

    parameters = [
        BeamParam('gitlab_url', str, 'https://gitlab.dt.local', 'GitLab URL'),
        BeamParam('gitlab_token', str, 'your_gitlab_token', 'GitLab Token'),
        BeamParam('branch', str, 'main', 'Branch'),
        BeamParam('beam_dir', str, '/home/dayosupp/projects/beamds', 'Beam Directory'),
        BeamParam('ci_registry', str, 'harbor.dt.local', 'CI Registry'),
        BeamParam('config_file', str, 'config.yaml', 'Config File'),
        BeamParam('commit_message', str, 'Update CI/CD pipeline configuration', 'Commit Message'),
        BeamParam('file_path', str, '.gitlab-ci.yml', 'File Path'),
        BeamParam('git_namespace', str, 'dayosupp', 'Namespace'),
        BeamParam('gitlab_project', str, 'dayosupp/yolo', 'GitLab Project'),
        BeamParam('commit_message', str, 'Update CI/CD pipeline configuration', 'Commit Message'),
        BeamParam('image_name', str, 'harbor.dt.local/public/beam:20240801', 'Image Name'),
        BeamParam('registry_user', str, 'admin', 'Registry User'),
        BeamParam('registry_url', str, 'https://harbor.dt.local', 'Registry URL'),
        BeamParam('registry_name', str, 'Registry Name'),
        BeamParam('registry_password', str, 'Har@123', 'Registry Password'),
        BeamParam('python_script', str, 'cicd_runner.py', 'Python Script'),
        BeamParam('cmd', str, 'python3', 'Command'),
        BeamParam('python_file', str, 'main.py', 'Python File where the algorithm is defined'),
        BeamParam('python_function', str, 'main', 'Python Function in the Python File which builds the object/algorithm'),
        BeamParam('bash_script', str, 'run_yolo.sh', 'Bash Script'),
        BeamParam('pipeline_tags', list, ['shell'], 'Pipeline Tags'),
        BeamParam('working_dir', str, '/app', 'Working Directory'),
        BeamParam('ssl_verify', bool, False, 'SSL Verify'),
        BeamParam('stages', list, ['build', 'test', 'deploy', 'release', 'run', 'deploy', 'before_script'], 'Stages'),

    ]

class ServeCICDConfig(BeamCICDConfig, ServeClusterConfig):
    pass
