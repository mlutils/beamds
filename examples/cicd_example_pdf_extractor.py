from ray.serve.scripts import deploy

from beam.git import BeamCICDClient, ServeCICDConfig
from beam import deploy_server
from beam.resources import resource
from beam.logging import beam_logger as logger


# Example usage of BeamCICD classs
def example_create_cicd_pipeline():
    # api_url, api_token, git_namespace = 'https://gitlab.dt.local', 'glpat-_fKCXzehNxPP3Do8QRx-', 'dayosupp'


    # conf = BeamCICDConfig(gitlab_url=api_url, gitlab_token=api_token, git_namespace=git_namespace)
    base_conf = ServeCICDConfig()
    yaml_conf = ServeCICDConfig(resource('/home/dayosupp/projects/beamds/examples/cicd_example_pdf_extractor.yaml').read())
    config = ServeCICDConfig(**{**base_conf, **yaml_conf})
    beam_cicd = BeamCICDClient(config)

    # logger.info(f"the config is: {config}")
    # beam_cicd.create_run_pipeline()
    image_name = beam_cicd.create_build_pipeline()


    logger.info("Deploying bundle via manager...")
    logger.info(f"Updating config with image name: {image_name}")
    config.update({'alg': image_name, 'path_to_state': image_name})
    logger.info(f"the config is: {config}")
    # config.update(alg=config.path_to_state)

    manager = resource('http://api-35000-beam-manager-9xhrw-dev.apps.kh-dev.dt.local')
    manager.launch_serve_cluster(config)

    logger.info('Done!')


if __name__ == "__main__":
    example_create_cicd_pipeline()
