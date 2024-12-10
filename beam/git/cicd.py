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
            stages = self.get_hparam('stages')
            pipeline_tags = self.get_hparam('pipeline_tags')
            # Prepare .gitlab-ci.yml content using provided parameters
            ci_template = {
                'variables': {
                    'IMAGE_NAME': self.get_hparam('image_name'),
                    'BEAM_DIR': self.get_hparam('beam_dir'),
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
                'stages': stages,
                'before_script': [
                        'echo "CI_PROJECT_NAMESPACE is :" $CI_PROJECT_NAMESPACE',
                        'git reset --hard',
                        'git clean -xdf',
                        'echo "Starting run_yolo job..."' #Todo: replace with message parameters
                ],
                'run_yolo_script': {
                    'stage': stages[6],
                    'tags': [pipeline_tags[0]],
                    'script': [
                        'echo $REGISTRY_PASSWORD | docker login -u $REGISTRY_USER --password-stdin $REGISTRY_URL',
                        'echo "mount : " $WORKING_DIR/$(basename $PYTHON_SCRIPT)," image : ", $IMAGE_NAME, " cmd : ", $CMD, " working dir : ", $WORKING_DIR/$(basename $PYTHON_SCRIPT), " ci project dir : ", $CI_PROJECT_DIR',
                        'docker run --rm --gpus all --entrypoint "/bin/bash" -v "$CI_PROJECT_DIR:$WORKING_DIR" -v "$BEAM_DIR:$BEAM_DIR" "$IMAGE_NAME" -c "export PYTHONPATH=$BEAM_DIR:$PYTHONPATH && $CMD $WORKING_DIR/$(basename $PYTHON_SCRIPT)"'
                    ],
                    'only': [self.get_hparam('branch')]
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

                actions = []

                # Include cicd_runner.py as part of the commit
                logger.info(f"Preparing to include 'cicd_runner.py' in the commit.")
                with open(str(path_to_runner), 'r') as f:
                    cicd_runner_content = f.read()
                actions.append({
                    'action': 'create' if 'cicd_runner.py' not in file_paths else 'update',
                    'file_path': 'cicd_runner.py',
                    'content': cicd_runner_content
                })

                if file_path in file_paths:
                    # If the file exists, update it with a commit
                    logger.info(f"File '{file_path}' exists. Preparing to update it.")

                    actions.append({
                        'action': 'update',
                        'file_path': file_path,
                        'content': ci_yaml_content
                    })

                elif file_path not in file_paths:
                    # If the file does not exist, create it with a commit
                    logger.info(f"File '{file_path}' does not exist. Preparing to create it.")

                    actions.append({
                        'action': 'create',
                        'file_path': file_path,
                        'content': ci_yaml_content
                    })

                # Commit both .gitlab-ci.yml and cicd_runner.py in one go
                commit_data = {
                    'branch': self.get_hparam('branch'),
                    'commit_message': self.get_hparam('commit_message'),
                    'actions': actions
                }

                logger.info(f"Committing and pushing changes...")
                project.commits.create(commit_data)

                # pipeline_data = {
                #     'ref': self.get_hparam('branch'),
                # }
                #
                # pipeline = project.pipelines.create(pipeline_data)
                # logger.info(
                #     f"Pipeline triggered for branch '{self.get_hparam('branch')}'. Pipeline ID: {pipeline.id}")
            except Exception as e:
                logger.error(f"Failed to create or update CI/CD pipeline: {str(e)}")
                raise
        except Exception as e:
            logger.error(f"Failed to create or update CI/CD pipeline: {str(e)}")
            raise