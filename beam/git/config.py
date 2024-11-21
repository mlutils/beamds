from ..config import BeamConfig, BeamParam

class BeamCICDConfig(BeamConfig):

    parameters = [

        BeamParam('gitlab_url', str, 'https://gitlab.dt.local', 'GitLab URL'),
        BeamParam('gitlab_token', str, 'your_gitlab_token', 'GitLab Token'),
        BeamParam('branch', str, 'main', 'Branch'),
        BeamParam('namespace', str, 'your_namespace', 'Namespace'),
        BeamParam('GITLAB_PROJECT', str, 'researchers/yolo_project', 'GitLab Project'),
        BeamParam('IMAGE_NAME', str, 'harbor.dt.local/public/beam:20240801', 'Image Name'),
        BeamParam('REGISTRY_USER', str, 'Registry User'),
        BeamParam('REGISTRY_PASSWORD', str, 'Registry Password'),
        BeamParam('PYTHON_FILE', str, 'main.py', 'Python File'),
        BeamParam('PYTHON_FUNCTION', str, 'main', 'Python Function'),
        BeamParam('BASH_SCRIPT', str, 'run_yolo.sh', 'Bash Script'),
        BeamParam('WORKING_DIR', str, '/opt/yolo', 'Working Directory'),

    ]
