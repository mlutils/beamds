import yaml
import gitlab
from urllib.parse import urlparse
from pathlib import Path
from ..path import beam_path
from ..utils import cached_property
from ..logging import beam_logger as logger
from ..base import BeamBase


class BeamCICD(BeamBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gitlab_url = self.get_hparam('gitlab_url')
        self.gitlab_token = self.get_hparam('gitlab_token')

    @cached_property
    def gitlab_client(self):
        return gitlab.Gitlab(self.gitlab_url, private_token=self.gitlab_token, ssl_verify=False)

    def create_cicd_pipeline(self, config=None):
        """
        Create a GitLab CI/CD pipeline configuration based on the provided parameters.

        config: Dictionary containing configuration like GITLAB_PROJECT, IMAGE_NAME, etc.
        @param config:
        """

        current_dir = beam_path(__file__).parent
        path_to_runner = current_dir.joinpath('cicd_runner.py')

        try:
            # Retrieve GitLab project
            project = self.gitlab_client.projects.get(self.get_hparam('gitlab_project'))

            # Prepare .gitlab-ci.yml content using provided parameters
            ci_template = {
                'variables': {
                    'IMAGE_NAME': self.get_hparam('image_name'),
                    'REGISTRY_USER': self.get_hparam('registry_user'),
                    'REGISTRY_PASSWORD': self.get_hparam('registry_password'),
                    'CI_REGISTRY': self.get_hparam('ci_registry'),
                    'PYTHON_FILE': self.get_hparam('python_file'),
                    'PYTHON_FUNCTION': self.get_hparam('python_function'),
                    'PYTHON_SCRIPT': path_to_runner.str,
                    'WORKING_DIR': self.get_hparam('working_dir'),
                    'CMD': self.get_hparam('cmd')
                },
                'stages': [
                    'run'
                ],
                'before_script': [
                    'echo "Starting run_yolo job..."' #Todo: replace with message parameters
                ],
                'run_yolo_script': {
                    'stage': 'run',
                    'tags': ['shell'],
                    'script': [
                        'echo "$REGISTRY_PASSWORD" | docker login -u $REGISTRY_USER --password-stdin $CI_REGISTRY',
                        '# - docker pull $IMAGE_NAME',
                        'docker run --rm --gpus all --entrypoint "/bin/bash" -v "$CI_PROJECT_DIR:$WORKING_DIR" $IMAGE_NAME $CMD $WORKING_DIR/$PYTHON_SCRIPT'
                    ],
                    'only': ['main']
                }
            }

            # Convert CI/CD template to YAML format
            ci_yaml_content = yaml.dump(ci_template)

            # Create or update the .gitlab-ci.yml file in the repository
            file_path = '.gitlab-ci.yml'
            try:
                # Try to fetch the file
                file = project.files.get(file_path=file_path, ref=self.get_hparam('branch'))
                # If the file exists, update it
                file.content = ci_yaml_content
                file.save(branch=self.get_hparam('branch'), commit_message='Update CI/CD pipeline configuration')
                logger.info(f"Updated .gitlab-ci.yml for project {self.get_hparam('gitlab_project')}")
            except gitlab.exceptions.GitlabGetError:
                # If the file doesn't exist, create it
                project.files.create({
                    'file_path': file_path,
                    'branch': self.get_hparam('branch'),
                    'content': ci_yaml_content,
                    'commit_message': 'Add CI/CD pipeline configuration'
                })
                logger.info(f"Created .gitlab-ci.yml for project {self.get_hparam('gitlab_project')}")

        except Exception as e:
            logger.error(f"Failed to create or update CI/CD pipeline: {str(e)}")
            raise


