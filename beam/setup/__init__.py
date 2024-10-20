# setup.py
import importlib
from termcolor import colored
import sys
import os


class BeamImporter:
    def __init__(self):
        self.modules = {}
        self.callables = {}
        self.aliases = {
            'pd': 'pandas',
            'np': 'numpy',
            'torch': 'torch',
            'F': 'torch.nn.functional',
            'Path': 'pathlib.Path',
            'plt': 'matplotlib.pyplot',
            # 'sns': 'seaborn',
            # 'tg': 'torch_geometric',
            'nn': 'torch.nn',
            'optim': 'torch.optim',
            'distributions': 'torch.distributions',
            'os': 'os',
            'sys': 'sys',
            'inspect': 'inspect',
            'time': 'time',
            'timedelta': 'datetime.timedelta',
            'random': 'random',
            'nx': 'networkx',
            're': 're',
            'glob': 'glob',
            'pickle': 'pickle',
            'json': 'json',
            'datetime': 'datetime.datetime',
            'date': 'datetime.date',
            'tqdm': 'tqdm.notebook.tqdm',
            'beam': 'beam',
            'defaultdict': 'collections.defaultdict',
            'Counter': 'collections.Counter',
            'OrderedDict': 'collections.OrderedDict',
            'partial': 'functools.partial',
            'namedtuple': 'collections.namedtuple',
            'reduce': 'functools.reduce',
            'itertools': 'itertools',
            'copy': 'copy',
            'warnings': 'warnings',
            'deepcopy': 'copy.deepcopy',
            'yaml': 'yaml',

        }
        self.__all__ = list(self.aliases.keys())  # Iterable of attribute names
        self.__file__ = None
        self._initialize_aliases()

    def _initialize_aliases(self):
        pass

    def __getattr__(self, name):
        if name in self.aliases:
            actual_name = self.aliases[name]
        else:
            actual_name = name

        try:
            imported_object = importlib.import_module(actual_name)
            # print(f"Info: actual_name: {actual_name}, imported_object: {imported_object}")
        except Exception as e:
            try:
                module_name, object_name = actual_name.rsplit('.', 1)
                module = importlib.import_module(module_name)
                imported_object = getattr(module, object_name)
                # print(f"Info: module_name: {module_name}, object_name: {object_name}, imported_object: {imported_object}")
            except:
                print(f"Warning: Could not import {actual_name}, {e}, Skipping...")
                # from ..utils import beam_traceback
                # print(beam_traceback())
                imported_object = None

        return imported_object


def load_ipython_extension(ipython, beam_path=None):
    import sys
    import os
    import time

    t0 = time.time()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    msg = colored('Setting up the Beam environment for interactive use', attrs=['bold'])
    print(f'✨ | {msg}')
    msg = colored('Standard modules will be automatically imported '
                  'so you can use them without explicit import', attrs=['bold'])
    print(f'🚀 | {msg}')

    if ipython is not None:
        ipython.run_line_magic('load_ext', 'autoreload')
        ipython.run_line_magic('autoreload', '2')

    beam_path = beam_path or os.getenv('BEAM_PATH', None)
    if beam_path is not None:
        sys.path.insert(0, beam_path)
        sys.path.insert(0, os.path.join(beam_path, 'src'))
    else:
        sys.path.insert(0, '..')
        sys.path.insert(0, '../src')

    beam_args = os.getenv('BEAM_ARGS', '')
    sys.argv.append('---BEAM-JUPYTER---')
    if beam_args:
        sys.argv.extend(beam_args.split(' '))

    for k in list(sys.modules.keys()):
        if k.startswith('beam'):
            del sys.modules[k]

    beam_importer = BeamImporter()
    msg = colored(f'Beam library is loaded from path: '
                  f'{os.path.abspath(os.path.dirname(beam_importer.beam.__file__))}', attrs=['bold'])
    print(f"🛸 | "f"{msg}")

    # Add the modules to the global namespace
    for alias in beam_importer.aliases:
        module = getattr(beam_importer, alias)
        if ipython is not None:
            ipython.push({alias: module})
        if alias == 'beam':
            for k in module.__all__:
                ipython.push({k: getattr(module, k)})

    msg = colored(f'Done importing packages. It took: {time.time() - t0: .2} seconds', attrs=['bold'])
    print(f"⏲ | {msg}")
    # print(f"The Beam version is: {beam_importer.beam.__version__}")


if __name__ == '__main__':

    beam_path = None
    if len(sys.argv) > 1:
        beam_path = sys.argv[1]
        # if not absolute path, make it absolute
        if not os.path.isabs(beam_path):
            beam_path = os.path.abspath(beam_path)

    load_ipython_extension(None, beam_path=beam_path)
