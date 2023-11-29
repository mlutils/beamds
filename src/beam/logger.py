import sys
import json
import loguru
from .utils import running_platform, pretty_format_number
from .path import beam_path
import atexit
import uuid
from datetime import datetime
import socket
import getpass
import traceback
from .path import beam_path
import time


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
            format = '{time:YYYY-MM-DD HH:mm:ss} ({elapsed}) | BeamLog | {level} | %s | {function} | {line} | {message}'\
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
        state = {'path': self.path.as_uri()}
        return state

    def __setstate__(self, state):
        self.__init__(state['path'])


beam_logger = BeamLogger()


def beam_kpi(beam_result_class, path=None):
    def _beam_kpi(func):
        def wrapper(x, *args, username=None, ip_address=None, algorithm=None, **kwargs):

            execution_time = datetime.now()

            # Get the IP address of the computer
            if ip_address is None:
                ip_address = socket.gethostbyname(socket.gethostname())

            # Get the username of the current user
            if username is None:
                username = getpass.getuser()

            algorithm_class = None
            algorithm_name = None
            experiment_path = None
            if algorithm is None:
                algorithm_class = type(algorithm).__name__
                if hasattr(algorithm, 'name'):
                    algorithm_name = algorithm.name
                if hasattr(algorithm, 'experiment') and algorithm.experiment is not None:
                    experiment_path = algorithm.experiment.experiment_dir

            result = None
            exception_message = None
            exception_type = None
            tb = None
            error = None
            try:
                with Timer() as timer:
                    result = func(x, *args, **kwargs)
            except Exception as e:
                error = e
                exception_message = str(e)
                exception_type = type(e).__name__
                tb = traceback.format_exc()
                beam_logger.exception(e)
            finally:

                metadata = dict(ip_address=ip_address, username=username, execution_time=execution_time,
                                elapsed=timer.elapsed, algorithm_class=algorithm_class, algorithm_name=algorithm_name,
                                experiment_path=experiment_path, exception_message=exception_message,
                                exception_type=exception_type, traceback=tb)

                logpaths = [path, kwargs.get('path')]

                kpi = beam_result_class(input=x, input_args=args, input_kwargs=kwargs, result=result,
                                        metadata=metadata, logpaths=logpaths)
                if error is not None:
                    raise error

                return kpi
        return wrapper
    return _beam_kpi


class BeamResult:

    def __init__(self, input=None, input_args=None, input_kwargs=None, result=None, metadata=None, logpaths=None):
        self.uuid = str(uuid.uuid4())
        self.input = input
        self.result = result
        self.metadata = metadata
        self.input_args = input_args
        self.input_kwargs = input_kwargs
        self.beam_logger = beam_logger
        if logpaths is None:
            logpaths = []
        logpaths = [path for path in logpaths if path is not None]
        self.logpaths = logpaths

        extra = {'type': 'kpi_metadata', 'uuid': {self.uuid}, 'input': self.input, 'input_args': self.input_args,
                 'input_kwargs': self.input_kwargs, 'result': self.result, **self.metadata}

        for logpath in self.logpaths:
            with self.beam_logger.open(logpath):
                self.beam_logger.info(f'BeamResult: {self.uuid} | username: {self.metadata["username"]} | '
                                      f'ip_address: {self.metadata["ip_address"]} | '
                                      f'execution_time: {self.metadata["execution_time"]} | '
                                      f'elapsed: {self.metadata["elapsed"]} | '
                                      f'algorithm_class: {self.metadata["algorithm_class"]} | '
                                      f'algorithm_name: {self.metadata["algorithm_name"]} | '
                                      f'experiment_path: {self.metadata["experiment_path"]} |'
                                      f'exception_message: {self.metadata["exception_message"]} | '
                                      f'exception_type: {self.metadata["exception_type"]} | ',
                                      extra=extra)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f'BeamResult(uuid={self.uuid}, input={self.input}, result={self.result}, metadata={self.metadata})'

    def like(self, explanation=None):

        extra = {'type': 'kpi_like', 'uuid': {self.uuid}, 'explanation': explanation}

        for logpath in self.logpaths:
            with self.beam_logger.open(logpath):

                self.beam_logger.info(f'KPI: {self.uuid} | like', extra=extra)
                if explanation is not None:
                    self.beam_logger.info(f'KPI: {self.uuid} | like | explanation: {explanation}', extra=extra)

    def dislike(self, explanation=None):

        extra = {'type': 'kpi_dislike', 'uuid': {self.uuid}, 'explanation': explanation}

        for logpath in self.logpaths:
            with self.beam_logger.open(logpath):
                self.beam_logger.warning(f'KPI: {self.uuid} | dislike', extra=extra)
                if explanation is not None:
                    self.beam_logger.warning(f'KPI: {self.uuid} | dislike | explanation: {explanation}', extra=extra)

    def rate(self, rating, explanation=None):

        extra = {'type': 'kpi_rate', 'uuid': {self.uuid}, 'rating': rating, 'explanation': explanation}

        if rating < 0 or rating > 5:
            raise ValueError('Rating must be between 0 and 5')

        if rating < 3:
            log_func = self.beam_logger.warning
        else:
            log_func = self.beam_logger.info

        for logpath in self.logpaths:
            with self.beam_logger.open(logpath):
                log_func(f'KPI: {self.uuid} | rating: {rating}/5', extra=extra)
                if explanation is not None:
                    log_func(f'KPI: {self.uuid} | rating: {rating}/5 | explanation: {explanation}', extra=extra)

    def notes(self, notes):

        extra = {'type': 'kpi_notes', 'uuid': {self.uuid}, 'notes': notes}
        for logpath in self.logpaths:
            with self.beam_logger.open(logpath):
                self.beam_logger.info(f'KPI: {self.uuid} | notes: {notes}', extra=extra)


class Timer(object):

    def __init__(self, name='', silence=False):
        self.name = name
        self.silence = silence
        self._elapsed = 0
        self.paused = True
        self.t0 = None

    def reset(self):
        self._elapsed = 0
        self.paused = True
        self.t0 = None
        return self

    def __enter__(self):
        if not self.silence:
            beam_logger.info(f"Starting timer: {self.name}")
        self.run()
        return self

    @property
    def elapsed(self):
        if self.paused:
            return self._elapsed
        return self._elapsed + time.time() - self.t0

    def pause(self):
        if self.paused:
            return self._elapsed
        self._elapsed = self._elapsed + time.time() - self.t0
        self.paused = True
        return self._elapsed

    def run(self):
        self.paused = False
        self.t0 = time.time()
        return self

    def __str__(self):
        return f"Timer {self.name}: state: {'paused' if self.paused else 'running'}, elapsed: {self.elapsed}"

    def __repr__(self):
        return str(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = self.pause()
        if not self.silence:
            beam_logger.info(f"Timer {self.name} paused. Elapsed time: {pretty_format_number(elapsed)} Sec")