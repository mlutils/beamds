import ast
import importlib
import sys
from ..path import beam_path


class ImportCollector(ast.NodeVisitor):
    def __init__(self):
        self.import_nodes = []

    def visit_Import(self, node):
        self.import_nodes.append(node)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        self.import_nodes.append(node)
        self.generic_visit(node)


def get_origin(module_name_or_spec):

    if type(module_name_or_spec) is str:
        module_name = module_name_or_spec

        try:
            spec = importlib.util.find_spec(module_name)
        except:
            return None

        if spec is None:
            return None

    else:
        spec = module_name_or_spec

    if hasattr(spec, 'origin') and spec.origin is not None:
        if spec.origin == 'built-in':
            return spec.origin
        origin = str(beam_path(spec.origin).resolve())
    else:
        try:
            origin = str(beam_path(spec.submodule_search_locations[0]).resolve())
        except:
            origin = None

    return origin


def get_module_paths(spec):

    if spec is None:
        return []

    paths = []

    if hasattr(spec, 'origin') and spec.origin is not None:
        origin = beam_path(spec.origin).resolve()
        if origin.is_file() and origin.parent.joinpath('__init__.py').is_file():
            origin = origin.parent

        paths.append(str(origin))

    if hasattr(spec, 'submodule_search_locations') and spec.submodule_search_locations is not None:
        for path in spec.submodule_search_locations:
            path = beam_path(path).resolve()
            if path.is_file():
                path = path.parent
            paths.append(str(path))

    return list(set(paths))


def classify_module(module_name):

    origin = get_origin(module_name)

    if origin == 'built-in':
        return "stdlib"

    if origin is None:
        return None

    # Get the standard library path using base_exec_prefix
    std_lib_path = beam_path(sys.base_exec_prefix).joinpath('lib')

    # Check for standard library
    if beam_path(origin).parts[:len(std_lib_path.parts)] == std_lib_path.parts:
        return "stdlib"

    # Check for installed packages in site-packages or dist-packages
    elif "site-packages" in origin or "dist-packages" in origin:
        return "installed"

    # Otherwise, it's a custom module
    else:
        return "custom"


def is_std_lib(module_name):
    return classify_module(module_name) == 'stdlib'


def is_installed_package(module_name):
    return classify_module(module_name) == 'installed'


def is_module_installed(module_name):
    try:
        importlib.metadata.version(module_name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False
