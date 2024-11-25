from beam.git import BeamCICD



# Example usage of BeamCICD class
def example_create_cicd_pipeline():
    api_url, api_token, namespace = 'https://gitlab.dt.local', 'git_token', 'your_namespace'
    beam_cicd = BeamCICD(gitlab_url=api_url, gitlab_token=api_token)

    config = {
        'GITLAB_PROJECT': 'researchers/yolo_project',
        'IMAGE_NAME': 'harbor.dt.local/public/beam:20240801',
        'REGISTRY_USER': 'admin',
        'REGISTRY_PASSWORD': 'Har@123',
        'CI_REGISTRY': 'harbor.dt.local'
    }

    beam_cicd.create_cicd_pipeline(config)


if __name__ == "__main__":
    example_create_cicd_pipeline()
