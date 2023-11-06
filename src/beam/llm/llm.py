from beam.llm.llm_models import OpenAIBase, TGILLM, FastChatLLM, FastAPILLM, HuggingFaceLLM
from beam.path import beam_key
import pandas as pd

import json
import numpy as np

from beam.utils import BeamURL
import openai
from typing import Any

from pydantic import PrivateAttr


class OpenAI(OpenAIBase):

    _models: Any = PrivateAttr()

    def __init__(self, model='gpt-3.5-turbo', api_key=None, organization=None, *args, **kwargs):

        api_key = beam_key('OPENAI_API_KEY', api_key)

        kwargs['scheme'] = 'openai'
        super().__init__(api_key=api_key, api_base='https://api.openai.com/v1',
                         organization=organization, *args, **kwargs)

        self.model = model
        self._models = None

    @property
    def is_chat(self):
        chat_models = ['gpt-4', 'gpt-4-0314', 'gpt-4-32k', 'gpt-4-32k-0314', 'gpt-3.5-turbo', 'gpt-3.5-turbo-0301']
        if any([m in self.model for m in chat_models]):
            return True
        return False

    def file_list(self):
        return openai.File.list()

    def build_dataset(self, data=None, question=None, answer=None, path=None) -> object:
        """
        Build a dataset for training a model
        :param data: dataframe with prompt and completion columns
        :param question: list of questions
        :param answer: list of answers
        :param path: path to save the dataset
        :return: path to the dataset
        """
        if data is None:
            data = pd.DataFrame(data={'prompt': question, 'completion': answer})

        records = data.to_dict(orient='records')

        if path is None:
            print('No path provided, using default path: dataset.jsonl')
            path = 'dataset.jsonl'

        # Open a file for writing
        with open(path, 'w') as outfile:
            # Write each data item to the file as a separate line
            for item in records:
                json.dump(item, outfile)
                outfile.write('\n')

        return path

    def retrieve(self, model=None):
        if model is None:
            model = self.model
        return openai.Engine.retrieve(id=model)

    @property
    def models(self):
        if self._models is None:
            models = openai.Model.list()
            models = {m.id: m for m in models.data}
            self._models = models
        return self._models

    def embedding(self, text, model=None):
        if model is None:
            model = self.model
        response = openai.Engine(model).embedding(input=text, model=model)
        embedding = np.array(response.data[1]['embedding'])
        return embedding


def beam_llm(url, username=None, hostname=None, port=None, api_key=None, **kwargs):

    if type(url) != str:
        return url

    url = BeamURL.from_string(url)

    if url.hostname is not None:
        hostname = url.hostname

    if url.port is not None:
        port = url.port

    if url.username is not None:
        username = url.username

    query = url.query
    for k, v in query.items():
        kwargs[k] = v

    if api_key is None and 'api_key' in kwargs:
        api_key = kwargs.pop('api_key')

    model = url.path
    model = model.strip('/')
    if not model:
        model = None

    if url.protocol == 'openai':

        api_key = beam_key('OPENAI_API_KEY', api_key)
        return OpenAI(model=model, api_key=api_key, **kwargs)

    elif url.protocol == 'fastchat':
        return FastChatLLM(model=model, hostname=hostname, port=port, **kwargs)

    elif url.protocol == 'huggingface':
        return HuggingFaceLLM(model=model, **kwargs)

    elif url.protocol == 'fastapi':
        return FastAPILLM(model=model, hostname=hostname, port=port, username=username, **kwargs)

    elif url.protocol == 'tgi':
        return TGILLM(model=model, hostname=hostname, port=port, username=username, **kwargs)

    else:
        raise NotImplementedError


import re


def text_splitter(text, separators=["\n\n", ". ", " "], chunk_size=100, length_function=None):
    if length_function is None:
        length_function = lambda x: int(1.5 * len(re.findall(r'\w+', x)))

    s = separators[0]
    open_chunks = text.split(s)
    closed_chunks = []
    next_chunk = ''

    for c in open_chunks:
        if length_function(c) > chunk_size:
            if len(next_chunk) > 0:
                closed_chunks.append(next_chunk)
            closed_chunks.extend(text_splitter(c, separators[1:], chunk_size, length_function))
            next_chunk = closed_chunks.pop()
        elif length_function(next_chunk) + length_function(c) > chunk_size:
            closed_chunks.append(next_chunk)
            next_chunk = c
        else:
            next_chunk = f"{next_chunk}{s}{c}"

    closed_chunks.append(next_chunk)

    return closed_chunks

