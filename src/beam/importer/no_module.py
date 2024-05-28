
class NoModule:

    def __init__(self, name):
        self.name = name

    def __getattr__(self, item):
        raise ImportError(f"Module '{self.name}' was not found, you should install it to use this feature.")

