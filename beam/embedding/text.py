from functools import cached_property


from ..utils import as_numpy, as_tensor, beam_device
from ..base import BeamResource


class BeamEmbedding(BeamResource):

    def encode(self, x):
        raise NotImplementedError

    def get_sentence_embedding_dimension(self):
        raise NotImplementedError

    def embed_documents(self, documents):
        return self.encode(documents)

    def embed_query(self, query):
        r = self.encode([query])
        return r[0]


class OpenAIEmbedding(BeamEmbedding):

    def __init__(self, model=None, api_key=None, api_base=None, tls=False, organization=None, format='tensor',
                 device=None, *args, **kwargs):
        super().__init__(*args, model=model, **kwargs)

        self.api_key = api_key
        if not api_base:
            self.api_base = None
        else:
            self.api_base = f"{'https' if tls else 'http'}://{api_base}"

        self.organization = organization
        self.model = model
        self.format = format
        self.device = device
        super().__init__(scheme='emb-openai', resource_type='encoder', **kwargs)

    def as_something(self, x):
        if self.format == 'numpy':
            return as_numpy(x)
        return as_tensor(x, device=self.device)

    @cached_property
    def client(self):
        from openai import OpenAI
        http_client = None
        if self.api_base:
            import httpx
            http_client = httpx.Client(transport=httpx.HTTPTransport(verify=False))

        return OpenAI(organization=self.organization, api_key=self.api_key, base_url=self.api_base,
                      http_client=http_client)

    def encode(self, x):
        res = self.client.embeddings.create(model=self.model, input=x, encoding_format='float')
        d = [res.data[i].embedding for i in range(len(res.data))]
        return self.as_something(d)


class SentenceTransformerEmbedding(BeamEmbedding):

    def __init__(self, model=None, device=None, batch_size=1, format='tensor', show_progress_bar=False, **kwargs):

        super().__init__(scheme='emb-stt', resource_type='encoder', **kwargs)

        self.device = beam_device(device)
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.convert_to_tensor = format == 'tensor'
        self.model = model
        self.network = self.load_model(model=model, device=self.device, **kwargs)

    @staticmethod
    def load_model(model=None, device=None, **st_kwargs):

        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model, device=device, **st_kwargs)

    def encode(self, x, batch_size=None, show_progress_bar=None):

        batch_size = batch_size or self.batch_size
        show_progress_bar = show_progress_bar or self.show_progress_bar
        convert_to_numpy = not self.convert_to_tensor

        if isinstance(x, str):
            x = [x]

        return self.network.encode(x, batch_size=batch_size, show_progress_bar=show_progress_bar,
                                   convert_to_numpy=convert_to_numpy, convert_to_tensor=self.convert_to_tensor)


