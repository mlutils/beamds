from .models import OpenAI, TGILLM, FastChatLLM, FastAPILLM, HuggingFaceLLM
from ..path import beam_key, BeamURL


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