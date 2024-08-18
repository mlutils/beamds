from typing import List
import torch

from ..logging import beam_logger as logger
from ..type.utils import Types
from ..utils import to_device, check_type, tqdm_beam as tqdm, divide_chunks, as_list
from ..resources import resource
from ..transformer import Transformer


class RobustDenseEncoder(Transformer):

    def __init__(self, encoder, device=None, batch_size=None, batch_ratios_to_try=None, **kwargs):
        super().__init__(batch_size=batch_size, **kwargs)
        self.device = device or 'cpu'
        self.encoder = encoder
        self.batch_size = self.get_hparam('batch_size', 32)
        self.batch_ratios_to_try = batch_ratios_to_try or [1, 4, 16, 32]

    def transform_callback(self, x, _key=None, _is_chunk=False, _fit=False, path=None, _store=False,
                           batch_size=None, show_progress_bar=True, **kwargs):

        batch_size = batch_size or self.batch_size
        for b in self.batch_ratios_to_try:
            b = max(1, batch_size // b)
            enc = self.encoder.encode(as_list(x), batch_size=b, show_progress_bar=show_progress_bar,
                                      convert_to_tensor=True)
            enc_type = check_type(enc)
            if enc_type.minor == Types.tensor:
                return to_device(enc, self.device)
            else:
                logger.warning(f"Encoding ({_key}) failed with batch size {b}. Retrying with smaller batch size.")

        logger.warning(f"Encoding ({_key}) failed with all batch sizes, defaulting to iterative calculation.")
        enc = []
        enci = self.encoder.encode('hi there', batch_size=1, show_progress_bar=show_progress_bar,
                                   convert_to_tensor=True)
        for xi in tqdm(x, enable=show_progress_bar):
            try:
                enci = self.encoder.encode(xi, batch_size=1, show_progress_bar=show_progress_bar,
                                           convert_to_tensor=True)
                enc.append(enci)
            except Exception as e:
                logger.error(f"Encoding failed for {xi}. {e}")
                enc.append(torch.zeros_like(enci))

        enc = torch.stack(enc)

        return enc

    def encode(self, x: List[str], batch_size=None, cache_dir=None, aggregate=True, **kwargs):
        batch_size = batch_size or self.batch_size
        cache_dir = resource(cache_dir)

        transform_params = dict(reduce=aggregate, return_results=True)
        if cache_dir is not None:
            transform_params['use_cache'] = True
            transform_params['store_path'] = cache_dir

        return self.transform(x, batch_size=batch_size, transform_params=transform_params)
