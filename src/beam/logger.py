import structlog
import logging.handlers

# Set up standard output logger with INFO and up messages
stdout_logger = structlog.configure(
    logger_factory=structlog.PrintLoggerFactory(),
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    level=logging.INFO,
)

# Set up text file logger with DEBUG and up textual messages
text_logger = structlog.configure(
    logger_factory=structlog.PrintLoggerFactory(),
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.KeyValueRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    level=logging.DEBUG,
    handlers=[logging.FileHandler("output.log")],
)

# Set up NDJSON file logger with JSON messages and stored struct
ndjson_logger = structlog.configure(
    logger_factory=structlog.PrintLoggerFactory(),
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(serializer=structlog.processors.KeyValueRenderer()),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    level=logging.DEBUG,
    handlers=[logging.handlers.RotatingFileHandler("output.ndjson", maxBytes=1000000, backupCount=5)],
)

# Use the loggers to write messages
stdout_logger.info("This message goes to standard output")
text_logger.debug("This message goes to a text file")
ndjson_logger.debug("This message goes to an NDJSON file", key1="value1", key2="value2")
