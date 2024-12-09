from beam.git import BeamCICD, BeamCICDConfig
from beam.resources import resource
from beam.logging import beam_logger as logger


# Example usage of BeamCICD classs
def example_create_cicd_pipeline():
    api_url, api_token, git_namespace = 'https://gitlab.dt.local', 'glpat-_fKCXzehNxPP3Do8QRx-', 'dayosupp'


    # conf = BeamCICDConfig(gitlab_url=api_url, gitlab_token=api_token, git_namespace=git_namespace)
    conf = BeamCICDConfig(resource('/home/dayosupp/projects/beamds/examples/cicd_example.yaml').read())
    beam_cicd = BeamCICD(conf)

    beam_cicd.create_cicd_pipeline()


if __name__ == "__main__":
    example_create_cicd_pipeline()
