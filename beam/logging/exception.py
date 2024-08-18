from .core import beam_logger as logger
from ..utils import beam_traceback


class BeamError(Exception):
    def __init__(self, message, error=None, **extra):

        traceback = None
        if error is not None:
            traceback = beam_traceback(error)

        logger.error(message, traceback=traceback, error=str(error), **extra)
        super().__init__(message)
