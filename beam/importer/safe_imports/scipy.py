from ..no_module import NoModule

try:
    import scipy
except ImportError:
    scipy = NoModule('scipy')