import sys
import loguru
from ..utils import running_platform
from ..path import beam_path
import atexit


class BeamLogger:

    def __init__(self, path=None, print=True):
        self.logger = loguru.logger
        self.logger.remove()
        self.handlers_queue = []
        self.running_platform = running_platform()

        self.handlers = {}
        if print:
            self.print()

        self.file_objects = {}
        self.path = None
        if path is not None:
            self.add_file_handlers(path)

        atexit.register(self.cleanup)

    def dont_print(self):
        self.logger.remove(self.handlers['stdout'])

    def print(self):
        self.handlers['stdout'] = self.logger.add(sys.stdout, level='INFO', colorize=True, format=
        '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | BeamLog | <level>{level}</level> | <level>{message}</level>')

    def cleanup(self, print=True):
        for k, handler in self.handlers.items():
            if k == 'stdout' and print:
                continue
            try:
                self.logger.remove(handler)
            except ValueError:
                pass

        if print:
            self.handlers = {k: v for k, v in self.handlers.items() if k == 'stdout'}
        else:
            self.handlers = {}

        for k, file_object in self.file_objects.items():
            file_object.close()
        self.file_objects = {}

    def add_file_handlers(self, path):

        path = beam_path(path)

        debug_path = path.joinpath('debug.log')
        file_object = debug_path.open('w')
        self.file_objects[str(debug_path)] = file_object

        if self.running_platform == 'script':
            format = '{time:YYYY-MM-DD HH:mm:ss} ({elapsed}) | BeamLog | {level} | {file} | {function} | {line} | {message}'
        else:
            format = '{time:YYYY-MM-DD HH:mm:ss} ({elapsed}) | BeamLog | {level} | %s | {function} | {line} | {message}' \
                     % self.running_platform

        handler = self.logger.add(file_object, level='DEBUG', format=format)

        self.handlers[str(debug_path)] = handler

        json_path = path.joinpath('json.log')
        file_object = json_path.open('w')
        self.file_objects[str(json_path)] = file_object

        format = 'JSON LOGGER'
        handler = self.logger.add(file_object, level='DEBUG', format=format, serialize=True)

        self.handlers[str(json_path)] = handler

    def open(self, path):
        path = beam_path(path)
        self.handlers_queue.append(path)
        return self

    def __enter__(self):
        self.add_file_handlers(self.handlers_queue[-1])
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        path = self.handlers_queue.pop()
        self.remove_file_handler(path)

    def remove_file_handler(self, name):
        for suffix in ['debug.log', 'json.log']:
            fullname = str(name.joinpath(suffix))
            self.logger.remove(self.handlers[fullname])
            self.handlers.pop(fullname)

    def debug(self, message, **extra):
        self.logger.debug(message, **extra)

    def info(self, message, **extra):
        self.logger.info(message, **extra)

    def warning(self, message, **extra):
        self.logger.warning(message, **extra)

    def error(self, message, **extra):
        self.logger.error(message, **extra)

    def critical(self, message, **extra):
        self.logger.critical(message, **extra)

    def exception(self, message, **extra):
        self.logger.exception(message, **extra)

    def __getstate__(self):

        if self.path is None:
            path = None
        elif isinstance(self.path, str):
            path = self.path
        else:
            path = self.path.as_uri()
        state = {'path': path}
        return state

    def __setstate__(self, state):
        self.__init__(state['path'])


beam_logger = BeamLogger()
