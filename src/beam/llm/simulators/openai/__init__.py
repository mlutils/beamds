from argparse import Namespace

from ...resource import beam_llm


def simulate_openai_chat(model=None, **kwargs):
    llm = beam_llm(model) if type(model) == str else model
    return llm.chat_completion(**kwargs).openai_format


def simulate_openai_completion(model=None, **kwargs):
    llm = beam_llm(model) if type(model) == str else model
    return llm.completion(**kwargs).openai_format


class OpenAI:

    chat = Namespace(completion=Namespace(create=simulate_openai_chat))
    completion = Namespace(create=simulate_openai_completion)

    def __init__(self, **kwargs):
       pass