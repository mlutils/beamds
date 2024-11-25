from beam.git import BeamCICD, BeamCICDConfig
from beam.logging import beam_logger as logger


# Example usage of BeamCICD class
def example_create_cicd_pipeline():
    api_url, api_token, namespace = 'https://gitlab.dt.local', 'glpat-_fKCXzehNxPP3Do8QRx-', 'dayosupp'


    conf = BeamCICDConfig(gitlab_url=api_url, gitlab_token=api_token, git_namespace=namespace)
    beam_cicd = BeamCICD(conf)

    beam_cicd.create_cicd_pipeline()


if __name__ == "__main__":
    example_create_cicd_pipeline()
