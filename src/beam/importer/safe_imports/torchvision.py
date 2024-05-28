from ..no_module import NoModule

try:
    from torchvision import transforms
except ImportError:
    torch = NoModule('torchvision.transforms')
