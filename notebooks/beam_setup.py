# beam_setup.py
import importlib


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
            'tg': 'torch_geometric',
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
        except:
            module_name, object_name = actual_name.rsplit('.', 1)
            module = importlib.import_module(module_name)
            imported_object = getattr(module, object_name)

        return imported_object


def load_ipython_extension(ipython):
    import sys
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    if ipython is not None:
        ipython.run_line_magic('load_ext', 'autoreload')
        ipython.run_line_magic('autoreload', '2')

    beam_path = os.getenv('BEAM_PATH', None)
    if beam_path is not None:
        sys.path.insert(0, beam_path)
        sys.path.insert(0, os.path.join(beam_path, 'src'))
    else:
        sys.path.insert(0, '..')
        sys.path.insert(0, '../src')

    beam_importer = BeamImporter()

    # Add the modules to the global namespace
    for alias in beam_importer.aliases:
        module = getattr(beam_importer, alias)
        if ipython is not None:
            ipython.push({alias: module})
        if alias == 'beam':
            for k in module.__dict__.keys():
                if not k.startswith('_'):
                    ipython.push({k: module.__dict__[k]})

    print("Setting up the Beam environment for interactive use")
    print("Standard modules are imported automatically so you can use them without explicit import")
    print(f"Beam library is loaded from path: {os.path.dirname(beam_importer.beam.__file__)}")
    print(f"The Beam version is: {beam_importer.beam.__version__}")


if __name__ == '__main__':
    load_ipython_extension(None)