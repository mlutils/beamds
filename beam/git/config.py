from ..config import BeamConfig, BeamParam
from ..orchestration import ServeClusterConfig

class BeamCICDConfig(BeamConfig):

    parameters = [
        BeamParam('gitlab_url', str, 'https://gitlab.dt.local', 'GitLab URL'),
        BeamParam('gitlab_token', str, 'your_gitlab_token', 'GitLab Token'),
        BeamParam('base_image', str, 'python:3.8', 'Base Image'),
        BeamParam('branch', str, 'main', 'Branch'),
        BeamParam('beam_dir', str, '/home/dayosupp/projects/beamds', 'Beam Directory'),
        BeamParam('ci_registry', str, 'harbor.dt.local', 'CI Registry'),
        BeamParam('ci_registry_project', str, 'public', 'CI Registry Project'),
        BeamParam('cicd_type', str, 'cicd_runner.py', 'CICD Runner type'),
        BeamParam('cmd', str, 'python3', 'Command'),
        BeamParam('config_file', str, 'config.yaml', 'Config File'),
        BeamParam('commit_message', str, 'Update CI/CD pipeline configuration', 'Commit Message'),
        BeamParam('dest_dir', str, '/app', 'Destination Directory'),
        BeamParam('entrypoint', str, 'python3', 'Entrypoint'),
        BeamParam('file_path', str, '.gitlab-ci.yml', 'File Path'),
        BeamParam('git_aux_files', list, [], 'Git Files'),
        BeamParam('git_namespace', str, 'dayosupp', 'Namespace'),
        BeamParam('gitlab_project', str, 'dayosupp/yolo', 'GitLab Project'),
        BeamParam('commit_message', str, 'Update CI/CD pipeline configuration', 'Commit Message'),
        BeamParam('image_name', str, 'harbor.dt.local/public/beam:20240801', 'Image Name'),
        BeamParam('registry_user', str, 'admin', 'Registry User'),
        BeamParam('registry_url', str, 'https://harbor.dt.local', 'Registry URL'),
        BeamParam('registry_name', str, 'Registry Name'),
        BeamParam('registry_password', str, 'Har@123', 'Registry Password'),
        BeamParam('requirements_file', str, 'requirements.txt', 'Requirements File'),
        BeamParam('python_script', str, 'cicd_runner.py', 'Python Script'),
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
