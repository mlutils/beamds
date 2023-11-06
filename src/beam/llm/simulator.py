from argparse import Namespace

from .resource import beam_llm


def simulate_openai_chat(model=None, **kwargs):
    llm = beam_llm(model) if type(model) == str else model
    return llm.chat_completion(**kwargs).openai_format


def simulate_openai_completion(model=None, **kwargs):
    llm = beam_llm(model) if type(model) == str else model
    return llm.completion(**kwargs).openai_format


openai_simulator = Namespace(ChatCompletion=Namespace(create=simulate_openai_chat),
                             Completion=Namespace(create=simulate_openai_completion))
