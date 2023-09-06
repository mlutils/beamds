from .utils_all import *

try:
    import torch
    has_torch = True
except ImportError:
    has_torch = False

if has_torch:
    from .utils_ds import *
