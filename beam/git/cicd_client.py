import yaml
import gitlab
from urllib.parse import urlparse
from pathlib import Path
from ..path import beam_path
from ..utils import cached_property
from ..logging import beam_logger as logger
from ..base import BeamBase
from datetime import datetime


class BeamCICDClient(BeamBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gitlab_url = self.get_hparam('gitlab_url')
        self.gitlab_token = self.get_hparam('gitlab_token')

    @cached_property
    def gitlab_client(self):
        return gitlab.Gitlab(self.gitlab_url, private_token=self.gitlab_token, ssl_verify=False)

    def create_build_pipeline(self, config=None):
        """
        Create a GitLab build CI/CD pipeline configuration based on the provided parameters.
        Parameters
        builds a docker image, pushes it to the registry and deploys it to openshift
        """

        current_dir = beam_path(__file__).parent
        current_dir.joinpath('config.yaml').write(config)

        try:
            # Retrieve GitLab project
            project = self.gitlab_client.projects.get(self.get_hparam('gitlab_project'))
            stages = self.get_hparam('stages')
            pipeline_tags = self.get_hparam('pipeline_tags')


            # Prepare Dockerfile content dynamically
            dockerfile_template = (
                """
                ARG BASE_IMAGE
                FROM ${BASE_IMAGE}

                ARG WORKING_DIR
                
                RUN mkdir -p ${WORKING_DIR}
                
                ARG ENTRYPOINT_SCRIPT
                
                WORKDIR ${WORKING_DIR}

                ARG REQUIREMENTS_FILE
                COPY ${REQUIREMENTS_FILE} ${REQUIREMENTS_FILE}

                RUN pip install --no-cache-dir -r ${WORKING_DIR}/${REQUIREMENTS_FILE}

                ARG PYTHON_FILE
                
                COPY ${PYTHON_FILE}  ${WORKING_DIR}/

                ARG ENTRYPOINT
                COPY ${ENTRYPOINT_SCRIPT} ${WORKING_DIR}/entrypoint.sh


                RUN chmod +x ${WORKING_DIR}/${ENTRYPOINT_SCRIPT}
                
                ENV PATH="${WORKING_DIR}:$PATH"
                
                RUN echo "PATH is : $PATH"
                
                ENV ENTRYPOINT="${WORKING_DIR}/${ENTRYPOINT_SCRIPT}"
                
                RUN echo "ENTRYPOINT is : ${ENTRYPOINT}"
                
                ENTRYPOINT ["entrypoint.sh"]
                """
            )

            current_time = datetime.now().strftime("%d%m%Y_%H%M")
            # Construct the image name dynamically
            image_name = f"{self.get_hparam('ci_registry')}/{self.get_hparam('ci_registry_project')}/{self.get_hparam('gitlab_project')}:{current_time}"

            # Prepare .gitlab-ci.yml content using provided parameters
            ci_template = {
                'variables': {
                    'IMAGE_NAME': image_name,
                    'BASE_IMAGE': self.get_hparam('base_image'),
                    'BEAM_DIR': self.get_hparam('beam_dir'),
                    'REGISTRY_USER': self.get_hparam('registry_user'),
                    'REGISTRY_PASSWORD': self.get_hparam('registry_password'),
                    'REGISTRY_URL': self.get_hparam('registry_url'),
                    'CI_REGISTRY': self.get_hparam('ci_registry'),
                    'CI_REGISTRY_PROJECT': self.get_hparam('ci_registry_project'),
                    'ENTRYPOINT_SCRIPT': self.get_hparam('entrypoint'),
                    'REQUIREMENTS_FILE': self.get_hparam('requirements_file'),
                    'PYTHON_FILE': self.get_hparam('python_script'),
                    'WORKING_DIR': self.get_hparam('working_dir'),
                    'CONFIG_FILE': self.get_hparam('config_file'),
                    'CMD': self.get_hparam('cmd')
                },
                'stages': stages,
                'before_script': [
                        'echo "CI_PROJECT_NAMESPACE is :" $CI_PROJECT_NAMESPACE',
                        'git reset --hard',
                        'git clean -xdf',
                        ''
                        'echo "Starting build job..."'
                ],
                'build': {
                    'stage': stages[0],
                    'tags': [pipeline_tags[0]],
                    'script': [
                        'echo $REGISTRY_PASSWORD | docker login -u $REGISTRY_USER --password-stdin $REGISTRY_URL',
                        'docker build \
                            --build-arg BASE_IMAGE=$BASE_IMAGE \
                            --build-arg WORKING_DIR=$WORKING_DIR \
                            --build-arg REQUIREMENTS_FILE=$REQUIREMENTS_FILE \
                            --build-arg APP_FILES=$CI_PROJECT_DIR \
                            --build-arg PYTHON_FILE=$PYTHON_FILE \
                            --build-arg ENTRYPOINT_SCRIPT=$ENTRYPOINT_SCRIPT \
                            -t $IMAGE_NAME .',
                        'docker push $IMAGE_NAME'
                    ],
                    'only': [self.get_hparam('branch')]
                }
            }

            # Convert CI/CD template to YAML format
            ci_yaml_content = yaml.dump(ci_template)

            try:
                # Try to fetch the file tree to check if the file exists
                file_tree = project.repository_tree(ref=self.get_hparam('branch'))
                file_paths = [f['path'] for f in file_tree]

                actions = []

                # Include Dockerfile as part of the commit
                actions.append({
                    'action': 'create' if 'Dockerfile' not in file_paths else 'update',
                    'file_path': 'Dockerfile',
                    'content': dockerfile_template
                })

                # Include .gitlab-ci.yml as part of the commit
                actions.append({
                    'action': 'create' if '.gitlab-ci.yml' not in file_paths else 'update',
                    'file_path': '.gitlab-ci.yml',
                    'content': ci_yaml_content
                })

                # Commit both Dockerfile and .gitlab-ci.yml in one go
                commit_data = {
                    'branch': self.get_hparam('branch'),
                    'commit_message': 'Add or update CI/CD configuration',
                    'actions': actions
                }

                project.commits.create(commit_data)

            except Exception as e:
                logger.error(f"Failed to create or update CI/CD pipeline: {str(e)}")
                raise

            return image_name

        except Exception as e:
            raise RuntimeError(f"Error creating build pipeline: {e}")
        # todo: this function generate yaml file which describes the build process and push it to registry
        # pass

    def create_run_pipeline(self, config=None):
        """
        Create a GitLab running CI/CD pipeline configuration based on the provided parameters.

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
                cicd_runner_content = path_to_runner.read()
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