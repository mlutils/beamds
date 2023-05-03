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
    def wrapper(*args, **kwargs):

        # # Get the IP address of the computer
        # ip_address = socket.gethostbyname(socket.gethostname())
        #
        # # Get the username of the current user
        # username = getpass.getuser()
        #
        # print(f"IP Address: {ip_address}")
        # print(f"Username: {username}")

        execution_time = datetime.now()
        result = None
        e = None
        try:
            with Timer() as timer:
                result = func(*args, **kwargs)
        except Exception as e:
            beam_logger.exception(e)
        finally:
            kpi = BeamKPI(result=result, elapsed=timer.elapsed, exception=e)
            if e is not None:
                raise e
        return kpi
    return wrapper


class BeamKPI:

    def __init__(self, result=None, elapsed=None, exception=None):
        self.uuid = str(uuid.uuid4())

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

