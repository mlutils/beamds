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
        current_dir.joinpath('config.yaml').write(config)

        try:
            # Retrieve GitLab project
            project = self.gitlab_client.projects.get(self.get_hparam('gitlab_project'))

            # Prepare .gitlab-ci.yml content using provided parameters
            ci_template = {
                'variables': {
                    'IMAGE_NAME': self.get_hparam('image_name'),
                    'REGISTRY_USER': self.get_hparam('registry_user'),
                    'REGISTRY_PASSWORD': self.get_hparam('registry_password'),
                    'REGISTRY_URL': self.get_hparam('registry_url'),
                    'CI_REGISTRY': self.get_hparam('ci_registry'),
                    'PYTHON_FILE': self.get_hparam('python_file'),
                    'PYTHON_FUNCTION': self.get_hparam('python_function'),
                    'PYTHON_SCRIPT': path_to_runner.str,
                    'WORKING_DIR': self.get_hparam('working_dir'),
                    'CONFIG_FILE': self.get_hparam('config_file'),
                    'CMD': self.get_hparam('cmd')
                },
                'stages': [ self.get_hparam('stages')],
                'before_script': {
                    'stage': [self.get_hparam('stages')[5]],
                    'tags': [self.get_hparam('pipeline_tags:')[0]],
                    'script': [
                        'echo "CI_PROJECT_NAMESPACE is :" $CI_PROJECT_NAMESPACE',
                        'git reset --hard',
                        'git clean -xdf',
                        'echo "Starting run_yolo job..."' #Todo: replace with message parameters
                    ],
                    'only': [{self.get_hparam('branch')}]
                },
                'run_yolo_script': {
                    'stage': [self.get_hparam('stages')[6]],
                    'tags': [self.get_hparam('pipeline_tags:')[0]],
                    'script': [
                        'echo "$REGISTRY_PASSWORD" | docker login -u $REGISTRY_USER --password-stdin $REGISTRY_URL',
                        '# - docker pull $IMAGE_NAME',
                        'docker cp {path_to_runner} $WORKING_DIR/$PYTHON_SCRIPT',
                        'docker cp $CONFIG_FILE $WORKING_DIR/$CONFIG_FILE',
                        'docker run --rm --gpus all --entrypoint "/bin/bash" -v "$CI_PROJECT_DIR:$WORKING_DIR" $IMAGE_NAME $CMD $WORKING_DIR/$PYTHON_SCRIPT'
                    ],
                    'only': [{self.get_hparam('branch')}]
                }
            }

            # Convert CI/CD template to YAML format
            ci_yaml_content = yaml.dump(ci_template)

            # Create or update the .gitlab-ci.yml file in the repository
            file_path = self.get_hparam('file_path')
            try:
                # Try to fetch the file tree to check if the file exists
                file_tree = project.repository_tree(ref=self.get_hparam('branch'))
                file_paths = [f['path'] for f in file_tree]

                if file_path in file_paths:
                    # If the file exists, update it with a commit
                    logger.info(f"File '{file_path}' exists. Preparing to update it.")
                    commit_data = {
                        'branch': {self.get_hparam('branch')},
                        'commit_message': self.get_hparam('commit_message'),
                        'actions': [
                            {
                                'action': 'update',
                                'file_path': '.gitlab-ci.yml',
                                'content': ci_yaml_content
                            }
                        ]
                    }
                    project.commits.create(commit_data)
                    logger.info(f"Updated and committed {file_path} for project {self.get_hparam('gitlab_project')}")
                elif file_path not in file_paths:
                    # If the file does not exist, create it with a commit
                    logger.info(f"File '{file_path}' does not exist. Preparing to create it.")
                    commit_data = {
                        'branch': {self.get_hparam('branch')},
                        'commit_message': self.get_hparam('commit_message'),
                        'actions': [
                            {
                                'action': 'create',
                                'file_path': file_path,
                                'content': ci_yaml_content
                            }
                        ]
                    }
                    project.commits.create(commit_data)
                    logger.info(f"Created and committed {file_path} for project {self.get_hparam('gitlab_project')}")
            except Exception as e:
                logger.error(f"Failed to create or update CI/CD pipeline: {str(e)}")
                raise
        except Exception as e:
            logger.error(f"Failed to create or update CI/CD pipeline: {str(e)}")
            raise