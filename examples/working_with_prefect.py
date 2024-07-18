import numpy as np
import torch
from beam.workflow.core import beam_flow
from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
import datetime
from dataclasses import dataclass


from beam.processor import Processor
from beam.transformer import Transformer
from beam.utils import check_type
from beam import beam_path, beam_hash


class DummyTransformer(Transformer):
    def transform_callback(self, x, _key=None, _is_chunk=False, _fit=False, path=None, **kwargs):
        res = []
        for xi in x:
            res.append(check_type(xi))
        return res



if __name__ == '__main__':
    beam_flow(DummyTransformer(name='dummy_transformer'), 'transform_callback',
              x=[np.random.randn(100), ['asdf', 'dsf','erer'], torch.arange(10)])

