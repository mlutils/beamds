from functools import wraps
from collections import defaultdict

from ..utils import LimitedSizeDictFactory, beam_hash
from ..path import beam_path
from ..data import BeamData
from ..logging import beam_logger as logger


def beam_cache(depth=None, cache_path=None, exception_keys=None, store_suffix=None, silent=False):
    def decorator(func):

        _depth, _cache_path, _exception_keys, _store_suffix, _silent = (depth, cache_path, exception_keys, store_suffix,
                                                                        silent)

        @wraps(func)
        def wrapper(self, *args, **kwargs):

            nonlocal _depth, _cache_path, _exception_keys, _store_suffix, _silent
            func_name = func.__name__
            _exception_keys = _exception_keys or []

            if hasattr(self, 'hparams'):
                _depth = _depth or self.hparams.get('cache_depth', None)
                _cache_path = _cache_path or self.hparams.get('cache_path', None)
                _exception_keys.extend(self.hparams.get('cache_exception_keys', []))
                _store_suffix = _store_suffix or self.hparams.get('cache_store_suffix', None)
                _silent = _silent or self.hparams.get('silent_cache', False)

            def on_remove(key, value):
                if _cache_path is not None:
                    key_path = beam_path(_cache_path).joinpath(value)
                    key_path.unlink()

            if not hasattr(self, '_results_cache'):
                setattr(self, '_results_cache', defaultdict(LimitedSizeDictFactory(size_limit=_depth,
                                                                                   on_removal=on_remove)))

            kwargs_to_hash = {k: v for k, v in kwargs.items() if k not in _exception_keys}
            hash_key = beam_hash((args, kwargs_to_hash))

            if hash_key in self._results_cache[func_name]:
                if _cache_path is not None:
                    key_path = beam_path(_cache_path).joinpath(self._results_cache[func_name][hash_key])

                    if _store_suffix is not None:
                        if not _silent:
                            logger.info(f"Result is cached. Reading from file cache: {key_path}")

                        result = key_path.read()
                    else:
                        if not _silent:
                            logger.info(f"Result is cached. Reading from BeamData cache: {key_path}")
                        bd = BeamData.from_path(key_path)
                        bd.cache()
                        result = bd.values
                else:
                    if not _silent:
                        logger.info(f"Result is cached. Reading from memory cache.")
                    result = self._results_cache[func_name][hash_key]

            else:
                result = func(self, *args, **kwargs)

                if _cache_path is not None:

                    _cache_path = beam_path(_cache_path)
                    _cache_path.mkdir()

                    key_path = _cache_path.joinpath(f'{func_name}_{hash_key}')
                    if _store_suffix is not None:
                        key_path = key_path.with_suffix(_store_suffix)

                        if not _silent:
                            logger.info(f"Caching result to file: {key_path}")
                        key_path.write(result)
                    else:
                        if not _silent:
                            logger.info(f"Caching result to BeamData: {key_path}")
                        bd = BeamData(data=result, path=key_path)
                        bd.store()

                    self._results_cache[func_name][hash_key] = key_path.name
                else:
                    if not _silent:
                        logger.info(f"Caching result to memory.")
                    self._results_cache[func_name][hash_key] = result

            return result

        return wrapper

    return decorator
