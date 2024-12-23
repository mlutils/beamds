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

    # beam_cicd.create_run_pipeline()
    obj = beam_cicd.create_build_pipeline()


    logger.info("Deploying bundle via manager...")
    config.update(hparams={'alg': obj  })
    # config.update(alg=config.path_to_state)

    manager = resource(config.manager_url)
    manager.launch_serve_cluster(config)

    logger.info('Done!')


if __name__ == "__main__":
    example_create_cicd_pipeline()
