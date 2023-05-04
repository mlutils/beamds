import sys
import json
import loguru
from .utils import running_platform, Timer
from .path import beam_path
import atexit
import uuid
from datetime import datetime
import socket
import getpass
import traceback


class BeamLogger:

    def __init__(self, path=None):
        self.logger = loguru.logger
        self.logger.remove()
        self.running_platform = running_platform()

        self.handlers = {'stdout': self.logger.add(sys.stdout, level='INFO', colorize=True, format=
           '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | BeamLog | <level>{level}</level> | <level>{message}</level>')}

        self.file_objects = {}
        self.path = None
        if path is not None:
             self.add_file_handlers(path)

        atexit.register(self.cleanup)

    def cleanup(self):
        for handler in self.handlers.values():
            self.logger.remove(handler)

        for file_object in self.file_objects.values():
            file_object.close()

    def add_file_handlers(self, path):

        self.path = path

        file_object = path.joinpath('debug.log').open('w')
        self.file_objects['debug'] = file_object

        if self.running_platform == 'script':
            format = '{time:YYYY-MM-DD HH:mm:ss} ({elapsed}) | BeamLog | {level} | {file} | {function} | {line} | {message}'
        else:
            format = '{time:YYYY-MM-DD HH:mm:ss} ({elapsed}) | BeamLog | {level} | %s | {function} | {line} | {message}'\
                     % self.running_platform

        handler = self.logger.add(file_object, level='DEBUG', format=format)

        self.handlers['debug'] = handler

        file_object = path.joinpath('json.log').open('w')
        self.file_objects['json'] = file_object

        format = 'JSON LOGGER'
        handler = self.logger.add(file_object, level='DEBUG', format=format, serialize=True)

        self.handlers['json'] = handler

    def remove_file_handler(self, name):
        self.logger.remove(self.handlers[name])
        self.handlers.pop(name)

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
        state = {'path': self.path.as_uri()}
        return state

    def __setstate__(self, state):
        self.__init__(state['path'])


beam_logger = BeamLogger()


def beam_kpi(func):
    def wrapper(x, *args, username=None, ip_address=None, algorithm=None, **kwargs):

        execution_time = datetime.now()

        # Get the IP address of the computer
        if ip_address is None:
            ip_address = socket.gethostbyname(socket.gethostname())

        # Get the username of the current user
        if username is None:
            username = getpass.getuser()

        if algorithm is None:
            algorithm_class = func.__name__

        result = None
        exception_message = None
        exception_type = None
        tb = None
        try:
            with Timer() as timer:
                result = func(x, *args, **kwargs)
        except Exception as e:
            exception_message = str(e)
            exception_type = type(e).__name__
            tb = traceback.format_exc()
            beam_logger.exception(e)
        finally:

            metadata = dict(ip_address=ip_address, username=username, execution_time=execution_time,
                            elapsed=timer.elapsed,
                            exception_message=exception_message, exception_type=exception_type, traceback=tb)

            kpi = BeamKPI(input=x, input_args=args, input_kwargs=kwargs, result=result, metadata=metadata)
            if e is not None:
                raise e
        return kpi
    return wrapper


class BeamKPI:

    def __init__(self, input=None, input_args=None, input_kwargs=None, result=None, metadata=None):
        self.uuid = str(uuid.uuid4())
        self.input = input
        self.result = result
        self.metadata = metadata
        self.input_args = input_args
        self.input_kwargs = input_kwargs

    def like(self, explanation=None):
        beam_logger.info('KPI: %s | like' % self.uuid)
        if explanation is not None:
            beam_logger.info('KPI: %s | explanation: %s' % (self.uuid, explanation))

    def dislike(self, explanation=None):
        beam_logger.info('KPI: %s | dislike' % self.uuid)

    def rate(self, rating, explanation=None):
        beam_logger.info('KPI: %s | rate: %s' % (self.uuid, rating))
        if explanation is not None:
            beam_logger.info('KPI: %s | explanation: %s' % (self.uuid, explanation))

    def notes(self, notes):
        beam_logger.info('KPI: %s | notes: %s' % (self.uuid, notes))

