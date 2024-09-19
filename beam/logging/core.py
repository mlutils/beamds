import sys
from contextlib import contextmanager
import loguru
import atexit


class BeamLogger:

    def __init__(self, path=None, print=True):
        self.logger = loguru.logger.opt(depth=1)
        self._level = None
        self.logger.remove()
        self.handlers_queue = []

        self.handlers = {}
        self.tags = {}
        if print:
            self.print()

        self.file_objects = {}
        self.path = None
        if path is not None:
            self.add_file_handlers(path)

        self.set_verbosity('INFO')

        atexit.register(self.cleanup)

    @property
    def running_platform(self):
        from ..utils import running_platform
        return running_platform()

    def dont_print(self):
        self.logger.remove(self.handlers['stdout'])

    def print(self):
        self.handlers['stdout'] = self.stdout_handler()

    def cleanup(self, print=True, clean_default=True, blacklist=None):
        if blacklist is None:
            blacklist = []
        for k, handler in self.handlers.items():
            if k == 'stdout' and print:
                continue
            if k in blacklist:
                continue
            if clean_default and k == 'default':
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

    @staticmethod
    def timestamp():
        import time
        t = time.strftime('%Y%m%d-%H%M%S')
        return t

    def add_default_file_handler(self, path):
        self.add_file_handlers(path, tag='default')

    def add_file_handlers(self, path, tag=None):
        from ..path import beam_path
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
        if tag is not None:
            self.tags[tag] = path

    def remove_tag(self, tag):
        path = self.tags[tag]
        self.remove_file_handler(path)

    def remove_default_handlers(self):
        self.remove_tag('default')

    def open(self, path):
        from ..path import beam_path
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

    def concat(self, messages):
        return ' '.join([str(m) for m in messages])

    def debug(self, *messages, **extra):
        self.logger.debug(self.concat(messages), **extra)

    def info(self, *messages, **extra):
        self.logger.info(self.concat(messages), **extra)

    def warning(self, *messages, **extra):
        self.logger.warning(self.concat(messages), **extra)

    def error(self, *messages, **extra):
        self.logger.error(self.concat(messages), **extra)

    def critical(self, *messages, **extra):
        self.logger.critical(self.concat(messages), **extra)

    def exception(self, *messages, **extra):
        self.logger.exception(self.concat(messages), **extra)

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

    def stdout_handler(self, level='INFO', file_info=True):

        file_info = f' <cyan>(∫{{file}}:{{function}}-#{{line}})</cyan>' if file_info else ''
        return self.logger.add(sys.stdout, level=level, colorize=True,
                               format=f'🔥 | <green>{{time:HH:mm:ss}} ({{elapsed}})</green> | '
                                      f'<level>{{level:<8}}</level> 🗎 <level>{{message}}</level>{file_info}')

    @property
    def level(self):
        return self._level

    def set_verbosity(self, level, file_info=True):
        """
        Sets the log level for all handlers to the specified level.
        """
        # Convert the level string to uppercase to match Loguru's expected levels
        level = level.upper()
        self._level = level

        if 'stdout' in self.handlers:
            self.logger.remove(self.handlers['stdout'])

        self.handlers['stdout'] = self.stdout_handler(level=level, file_info=file_info)

    def debug_mode(self, **kwargs):
        self.set_verbosity('DEBUG', **kwargs)
        self.debug('Debug mode activated')

    def info_mode(self, **kwargs):
        self.set_verbosity('INFO', **kwargs)
        self.info('Info mode activated')

    def warning_mode(self, **kwargs):
        self.set_verbosity('WARNING', **kwargs)
        self.warning('Warning mode activated (only warnings and errors will be logged)')

    def error_mode(self, **kwargs):
        self.set_verbosity('ERROR', **kwargs)
        self.error('Error mode activated (only errors will be logged)')

    def critical_mode(self, **kwargs):
        self.set_verbosity('CRITICAL', **kwargs)
        self.critical('Critical mode activated (only critical errors will be logged)')

    @contextmanager
    def as_debug_mode(self):
        mode = self.logger.level
        self.debug_mode()
        yield
        self.set_verbosity(mode)

    @contextmanager
    def as_info_mode(self):
        mode = self.logger.level
        self.info_mode()
        yield
        self.set_verbosity(mode)

    @contextmanager
    def as_warning_mode(self):
        mode = self.logger.level
        self.warning_mode()
        yield
        self.set_verbosity(mode)

    @contextmanager
    def as_error_mode(self):
        mode = self.logger.level
        self.error_mode()
        yield
        self.set_verbosity(mode)

    @contextmanager
    def as_critical_mode(self):
        mode = self.logger.level
        self.critical_mode()
        yield
        self.set_verbosity(mode)


beam_logger = BeamLogger()
