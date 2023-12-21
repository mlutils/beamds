from ..utils import Timer
import uuid
from datetime import datetime
import socket
import getpass
import traceback
from .core import beam_logger


def beam_kpi(beam_result_class, path=None):
    def _beam_kpi(func):
        def wrapper(x, *args, _username=None, _ip_address=None, _algorithm=None, **kwargs):

            execution_time = datetime.now()

            # Get the IP address of the computer
            if _ip_address is None:
                _ip_address = socket.gethostbyname(socket.gethostname())

            # Get the username of the current user
            if _username is None:
                _username = getpass.getuser()

            algorithm_class = None
            algorithm_name = None
            experiment_path = None
            if _algorithm is None:
                algorithm_class = type(_algorithm).__name__
                if hasattr(_algorithm, 'name'):
                    algorithm_name = _algorithm.name
                if hasattr(_algorithm, 'experiment') and _algorithm.experiment is not None:
                    experiment_path = _algorithm.experiment.experiment_dir

            result = None
            exception_message = None
            exception_type = None
            tb = None
            error = None
            try:
                with Timer(beam_logger) as timer:
                    result = func(x, *args, **kwargs)
            except Exception as e:
                error = e
                exception_message = str(e)
                exception_type = type(e).__name__
                tb = traceback.format_exc()
                beam_logger.exception(e)
            finally:

                metadata = dict(ip_address=_ip_address, username=_username, execution_time=execution_time,
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
