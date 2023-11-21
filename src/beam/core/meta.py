
class MetaLLM:
    pass


class MetaChain:
    pass


class PluginRunner:
    pass


class PluginAPI:
    pass





class MetaAlgoritm:
    """
    A meta algorithm which defines the interface for all algorithms. It is based on the sklearn interface.
    """

    def __init__(self, hparams, **kwargs):
        """
        The init function of the meta algorithm, requires a hparams object and an optional kwargs object
        @param hparams:
        @param kwargs:
        """
        self._hparams = hparams
        self._kwargs = kwargs
        self.state = None

        # This is an example of a property that is not set until it is called
        self._my_property = None

    def fit(self, X, y=None, **kwargs):
        """
        The fit function of the meta algorithm
        @param X:
        @param y:
        @return:
        """
        raise NotImplementedError()

    def predict(self, X, **kwargs):
        """
        The predict function of the meta algorithm
        @param X:
        @return:
        """
        raise NotImplementedError()

    def score(self, X, y=None, **kwargs):
        """
        The score function of the meta algorithm
        @param X:
        @param y:
        @return:
        """
        raise NotImplementedError()

    def save(self, path):
        """
        The save function of the meta algorithm
        @param path:
        @return:
        """
        raise NotImplementedError()

    def load(self, path):
        """
        The load function of the meta algorithm
        @param path:
        @return:
        """
        raise NotImplementedError()

    @property
    def my_property(self):
        """
        This function is an example of a property that is not set until it is called
        @return:
        """

        if self._my_property is None:
            self._my_property = 1

        return self._my_property

