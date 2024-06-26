from argparse import Namespace

from ...resource import beam_llm


def stream_openai_generator(response):
    for res in response:
        yield res.openai_format


def simulate_openai_chat(model=None, stream=False, **kwargs):
    llm = beam_llm(model) if type(model) == str else model
    res = llm.chat_completion(stream=stream, **kwargs)
    if stream:
        return stream_openai_generator(res)
    else:
        return res.openai_format


def simulate_openai_completion(model=None, **kwargs):
    llm = beam_llm(model) if type(model) == str else model
    return llm.completion(**kwargs).openai_format


class OpenAI:

    chat = Namespace(completions=Namespace(create=simulate_openai_chat))
    completions = Namespace(create=simulate_openai_completion)

    def __init__(self, **kwargs):
       pass