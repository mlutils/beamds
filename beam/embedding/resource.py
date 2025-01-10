from .text import OpenAIEmbedding, SentenceTransformerEmbedding
from ..path import normalize_host, beam_key, BeamURL

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
        kwargs[k.replace('-', '_')] = v

    if api_key is None and 'api_key' in kwargs:
        api_key = kwargs.pop('api_key')

    path = url.path

    if 'v1' in path:
        api_path, model = path.split('v1')
        api_path = f'{api_path}v1'
    else:
        model = path
        api_path = ''

    model = model.strip('/')
    if not model:
        model = None

    api_base = normalize_host(hostname, port)
    if api_path and api_base:
        api_base = f"{api_base}{api_path}"

    if hostname is None:
        api_base = None

    if url.protocol == 'emb-openai':
        api_key = beam_key('OPENAI_API_KEY', api_key)
        return OpenAIEmbedding(model=model, api_base=api_base, api_key=api_key, **kwargs)
    if url.protocol == 'emb-stt':
        return SentenceTransformerEmbedding(model=model, **kwargs)
    return None