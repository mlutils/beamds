from .text import OpenAIEmbedding, SentenceTransformerEmbedding
from ..base import BeamURL


def beam_embedding(url, username=None, hostname=None, port=None, api_key=None,  **kwargs):

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

    if url.protocol == 'emb-openai':
        return OpenAIEmbedding(model=model, hostname=hostname, port=port, api_key=api_key, **kwargs)
    if url.protocol == 'emb-stt':
        return SentenceTransformerEmbedding(model=model, **kwargs)
    return None