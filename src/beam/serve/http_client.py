import io

import requests
from ..utils import lazy_property
from .client import BeamClient


class HTTPClient(BeamClient):

    def __init__(self, *args, tls=False, **kwargs):
        from requests.packages.urllib3.exceptions import InsecureRequestWarning
        requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
        self.protocol = 'https' if tls else 'http'
        super().__init__(*args, **kwargs)

    def get_info(self):
        return requests.get(f'{self.protocol}://{self.host}/').json()

    def get(self, path, **kwargs):

        response = requests.get(f'{self.protocol}://{self.host}/{path}', **kwargs)
        if response.status_code == 200:
            response = self.load_function(io.BytesIO(response.content))

        return response

    def _post(self, path, io_args, io_kwargs, **other_kwargs):

        files = {'args': io_args, 'kwargs': io_kwargs}
        files = {**files, **other_kwargs}

        response = requests.post(f'{self.protocol}://{self.host}/{path}',
                                 files=files, stream=True)

        if response.status_code == 200:
            response = self.load_function(io.BytesIO(response.content))

        return response

