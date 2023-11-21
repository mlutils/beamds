
class BeamResource:
    """
    Base class for all resources. Gets as an input a URI and the resource type and returns the resource.
    """
    def __init(self, uri, resource_type):
        self.uri = uri
        self.resource_type = resource_type

    def as_uri(self):
        return self.uri