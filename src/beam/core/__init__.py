from .processor import Processor

try:
    import torch
    has_torch = True
except ImportError:
    has_torch = False

if has_torch:
    from .algorithm import Algorithm
