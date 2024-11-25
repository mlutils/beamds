import yaml
import gitlab

from beam.utils import cached_property
from beam.logging import beam_logger as logger
from beam.base import BeamBase


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
        try:
            # Retrieve GitLab project
            project = self.gitlab_client.projects.get(self.get_hparam('gitlab_project'))

            # Prepare .gitlab-ci.yml content using provided parameters
            ci_template = {
                'stages': ['run'],
                'variables': {
                    'IMAGE_NAME': self.get_hparam('image_name'),
                    'REGISTRY_USER': self.get_hparam('registry_user'),
                    'REGISTRY_PASSWORD': self.get_hparam('registry_password'),
                    'CI_REGISTRY': self.get_hparam('ci_registry'),
                    'PYTHON_FILE': self.get_hparam('python_file'),
                    'PYTHON_FUNCTION': self.get_hparam('python_function'),
                    'BASH_SCRIPT': self.get_hparam('bash_script'),
                    'WORKING_DIR': self.get_hparam('working_dir'),

                },
                'before_script': [
                    'echo "Starting run_yolo job..."'
                ],
                'run_yolo_script': {
                    'stage': 'run',
                    'tags': ['shell'],
                    'script': [
                        'echo "$REGISTRY_PASSWORD" | docker login -u $REGISTRY_USER --password-stdin $CI_REGISTRY',
                        '# - docker pull $IMAGE_NAME',
                        'docker run --rm --gpus all --entrypoint "/bin/bash" -v "$CI_PROJECT_DIR:$WORKING_DIR" $IMAGE_NAME $WORKING_DIR/$BASH_SCRIPT'
                    ],
                    'only': ['main']
                }
            }

            # Convert CI/CD template to YAML format
            ci_yaml_content = yaml.dump(ci_template)

            # Create or update the .gitlab-ci.yml file in the repository
            file_path = '.gitlab-ci.yml'
            try:
                file = project.files.get(file_path=file_path, ref='main')
                file.content = ci_yaml_content
                file.save(branch='main', commit_message='Update CI/CD pipeline configuration')
                logger.info(f"Updated .gitlab-ci.yml for project {config['GITLAB_PROJECT']}")

            except Exception as e:
                project.files.create({
                    'file_path': file_path,
                    'branch': 'main',
                    'content': ci_yaml_content,
                    'commit_message': 'Add CI/CD pipeline configuration'
                })
                logger.info(f"Created .gitlab-ci.yml for project {config['GITLAB_PROJECT']}")

            # Commit and push changes
            project.commits.create({
                'branch': self.get_hparam('branch'),
                'commit_message':  config.get('commit_message'),
                'actions': [{
                    'action': 'update',
                    'file_path': file_path,
                    'content': ci_yaml_content
                }]
            })
            logger.info(f"Committed and pushed changes for project {config['GITLAB_PROJECT']}")

        except Exception as e:
            logger.error(f"Failed to create or update CI/CD pipeline: {str(e)}")
            raise


