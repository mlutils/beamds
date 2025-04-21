import sys
from contextlib import contextmanager

import loguru
import atexit


class BeamLogger:

    def __init__(self, paths=None, print=True, colors=True):
        self.logger = loguru.logger.opt(depth=1)
        self._level = None
        self.colors = colors
        self.logger.remove()
        self.handlers_queue = []

        self.handlers = {}
        self.tags = {}
        if print:
            self.print()

        self.file_objects = {}
        self.paths = {}
        if type(paths) is dict:
            for t, p in paths.items():
                self.add_file_handlers(p, tag=t)
        elif type(paths) is str:
            self.add_file_handlers(paths, tag='default')
        elif type(paths) is list:
            for p in paths:
                self.add_file_handlers(p)

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

    @property
    def text_format(self):

        if self.running_platform == 'script':
            format = '{time:YYYY-MM-DD HH:mm:ss} ({elapsed}) | BeamLog | {level} | {file} | {function} | {line} | {message}'
        else:
            format = '{time:YYYY-MM-DD HH:mm:ss} ({elapsed}) | BeamLog | {level} | %s | {function} | {line} | {message}' \
                     % self.running_platform

        return format

    def add_file_handlers(self, path, tag=None):
        from ..path import beam_path
        path = beam_path(path)

        debug_path = path.joinpath('debug.log')
        file_object = debug_path.open('w')
        self.file_objects[file_object.as_uri()] = file_object

        handler = self.logger.add(file_object, level='DEBUG', format=self.text_format)

        self.handlers[file_object.as_uri()] = handler

        json_path = path.joinpath('json.log')
        file_object = json_path.open('w')
        self.file_objects[file_object.as_uri()] = file_object

        handler = self.logger.add(file_object, level='DEBUG', format='JSON LOGGER', serialize=True)

        self.handlers[file_object.as_uri()] = handler
        if tag is not None:
            self.tags[tag] = path
            self.paths[tag] = path
        else:
            self.paths[path.as_uri()] = path

    def add_directory_handler(
        self,
        directory,
        level: str = "DEBUG",
        fmt: str = None,
        *,
        with_extra: bool = False,
        serialize: bool = False
    ):
        """
        Log each record into its own file inside `directory`.  Filename is:
          {timestamp}_{level}_{file}_{function}_{line}.(log|json)

        :param with_extra: if True, write a dict containing _record, _ts, _level, etc.
        :param serialize:  if True, output JSON (record dict); otherwise plain text.
        """
        # 1. prepare path
        from ..path import beam_path
        directory = beam_path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # 2. decide format string
        if serialize:
            fmt = 'JSON LOGGER'
        else:
            fmt = fmt or self.text_format  # fall back to your existing text_format attribute

        # 3. define the per-record sink
        def _directory_sink(message):
            record = message.record
            ts = record["time"].strftime("%Y%m%d-%H%M%S-%f")[:-3]
            lvl = record["level"].name
            base = f"{ts}_{lvl}_{record['file'].name}_{record['function']}_{record['line']}"

            # choose extension & content
            if serialize:
                filename = f"{base}.json"
                # dump the full record + extras (note: message.output is ignored in JSON mode)
                content = {**record, "_ts": ts, "_level": lvl, "_filename": base, "_message": message}
            else:
                filename = f"{base}.log"
                # message.output is already formatted according to `fmt`
                # but we passed fmt to add() so message.output uses it
                if with_extra:
                    extra = {"_ts": ts, "_level": lvl, "_filename": base, "_record": record}
                    content = f"{message.output.strip()}  ║  {extra}\n"
                else:
                    content = message.output

            directory.joinpath(filename).write(content)

        # 4. register with Loguru
        handler_id = self.logger.add(
            _directory_sink,
            level=level,
            format=fmt,
            colorize=False,    # per-file logs usually shouldn’t have ANSI codes
            backtrace=False,
            diagnose=False,
        )
        # keep track so you can remove it later
        self.handlers[directory.as_uri()] = handler_id
        return handler_id

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

        paths = {k: v.as_uri() for k, v in self.paths.items()}
        state = {'paths': paths}
        return state

    def __setstate__(self, state):
        self.__init__(state['paths'])

    def stdout_handler(self, level='INFO', file_info=True, colors=True):

        file_info = f' <cyan>(∫{{file}}:{{function}}-#{{line}})</cyan>' if file_info else ''
        return self.logger.add(sys.stdout, level=level, colorize=colors,
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

        self.handlers['stdout'] = self.stdout_handler(level=level, file_info=file_info, colors=self.colors)

    def turn_colors_off(self, **kwargs):
        self.colors = False
        self.set_verbosity(self.level, **kwargs)
        self.debug('Colors in logs turned off')

    def turn_colors_on(self, **kwargs):
        self.colors = True
        self.set_verbosity(self.level, **kwargs)
        self.debug('Colors in logs turned on')

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

    def add_logstash(self, host, port=5044, version=1):
        """
        Adds a Logstash handler to send logs to a Logstash server.

        :param host: The host of the Logstash server.
        :param port: The port of the Logstash server (default: 5044).
        :param version: Logstash message format version (default: 1).
        """
        import logstash
        handler_name = 'logstash'

        if handler_name in self.handlers:
            self.debug("Logstash handler already exists. Skipping addition.")
            return

        logstash_handler = logstash.TCPLogstashHandler(host, port, version=version)
        self.handlers[handler_name] = self.logger.add(
            logstash_handler,
            level=self.level,
        )

        self.debug(f"Logstash handler added. Logs will be sent to {host}:{port}")


    def remove_logstash(self):
        """
        Removes the Logstash handler if it exists.
        """
        handler_name = 'logstash'

        if handler_name not in self.handlers:
            self.debug("Logstash handler not found. Nothing to remove.")
            return

        self.logger.remove(self.handlers[handler_name])
        del self.handlers[handler_name]

        self.debug("Logstash handler removed successfully.")


beam_logger = BeamLogger()
