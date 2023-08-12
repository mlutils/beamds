import inspect
import importlib
import ast
import os

from .tabular import DeepTabularAlg
from .path import beam_path

content = beam_path(inspect.getfile(DeepTabularAlg)).read()
ast_tree = ast.parse(content)
importlib.util.find_spec('torch.nn')

import pkg_resources
import os
import importlib

def get_pip_package(module_name):
    module = importlib.import_module(module_name)
    module_path = os.path.dirname(module.__file__)

    for dist in pkg_resources.working_set:
        package_path = dist.location
        if module_path.startswith(package_path):
            return f"{dist.project_name}=={dist.version}"

    return None

module_name = "fastchat"
pip_package_info = get_pip_package(module_name)

if pip_package_info:
    print(f"Add to requirements: {pip_package_info}")
else:
    print(f"No matching pip package found for module: {module_name}")
