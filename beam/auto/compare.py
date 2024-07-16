@staticmethod
def to_docker(obj=None, base_image='python:3.10-slim', image_name=None, beam_version=None,
              serve_config=None, registry_url=None, push_image=False, username=None, password=None, **kwargs):

    if obj is not None:
        logger.info(f"Building an object bundle")
        # Ensure the bundle path is handled inside to_bundle if not provided
        bundle_path = AutoBeam.to_bundle(obj)
    else:
        raise ValueError("Object must be provided for dockerization")

    logger.info(f"Building a Docker image with the requirements and the object bundle. Base image: {base_image}")
    # Assuming _build_image just prepares the image and does not return anything
    AutoBeam._build_image(bundle_path, base_image, config=serve_config, image_name=image_name,
                          beam_version=beam_version, **kwargs)

    if push_image:
        full_image_name = AutoBeam._push_image(image_name=image_name, registry_url=registry_url,
                                               username=username, password=password)
        return full_image_name  # Return the full image name including the tag
    else:
        # Return the local image name with a tag if push_image is False
        local_image_name = f"{image_name}:{beam_version if beam_version else 'latest'}"
        return local_image_name

@staticmethod
def to_bundle(obj, path=None):
    # Assume that the logic to resolve the path and handle bundling is correct
    if path is None:
        path = beam_path('.')
        if hasattr(obj, 'name'):
            path = path.joinpath(obj.name)
        else:
            path = path.joinpath('beam_bundle')
    path = beam_path(path).resolve()

    path.clean()
    path.mkdir()
    logger.info(f"Saving object's files to path {path}")
    # Simulate data serialization, e.g., saving requirements, modules etc.
    # This is where you'd serialize the object state, dependencies etc.
    return path
