import sys
import os
import importlib

try:
    from ..src import beam
    spec = importlib.util.find_spec('..src.beam')
except ImportError:
    try:
        from .src import beam
        spec = importlib.util.find_spec('.src.beam')
    except ImportError:
        try:
            from src import beam
            spec = importlib.util.find_spec('src.beam')
        except ImportError:
            import beam
            spec = importlib.util.find_spec('beam')


sys.path.insert(0, os.path.abspath(os.path.join(spec.origin, '../..')))

print(f"Beam library is loaded from path: {os.path.dirname(spec.origin)}")
print(f"The Beam version is: {beam.__version__}")
