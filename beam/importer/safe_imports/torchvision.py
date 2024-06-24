from ..no_module import NoModule

try:
    from torchvision import transforms
except ImportError:
    transforms = NoModule('torchvision.transforms')
