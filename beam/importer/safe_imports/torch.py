from ..no_module import NoModule

try:
    import torch
except ImportError:
    torch = NoModule('torch')

