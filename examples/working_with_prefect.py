import numpy as np
import torch
from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
import datetime
from dataclasses import dataclass


from src.beam.core import Processor
from src.beam.transformer import Transformer
from src.beam.utils import check_type
from src.beam import beam_path, beam_hash


class DummyTransformer(Transformer):
    def transform_callback(self, x, key=None, is_chunk=False, fit=False, path=None, **kwargs):
        res = []
        for xi in x:
            res.append(check_type(xi))
        return res



if __name__ == '__main__':
    beam_flow(DummyTransformer(name='dummy_transformer'), 'transform_callback',
              x=[np.random.randn(100), ['asdf', 'dsf','erer'], torch.arange(10)])

