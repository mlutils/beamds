import json
import re

from dataclasses import dataclass, field

from ..core import Processor
from ..path import beam_path, local_copy, BeamURL
from ..utils import lazy_property


@dataclass
class TritonConfig:
    name: str = ''
    platform: str = ''
    max_batch_size: int = 0
    input: list = field(default_factory=list)
    output: list = field(default_factory=list)
    instance_groups: list = field(default_factory=list)

    @staticmethod
    def transform_to_json_like(s):
        s = re.sub(r'(\w+)\s*:', r'"\1":', s)  # add quotes for keys
        s = re.sub(r'(\w+)\s+\[', r'"\1": [', s)  # add quotes for keys
        s = re.sub(r'(\w+)\s+{', r'"\1": {', s)  # add quotes for keys
        s = re.sub(r':\s*(?![\["\d])(\w+)', r': "\1"', s)  # add quotes to values
        s = re.sub(r'([}\]"])\s+("\w)', r'\1,\n\2', s)  # Add commas between elements
        s = re.sub(r'(\s+\d+)\s+("\w)', r'\1,\n\2', s)  # Add commas between elements
        s = f"{{{s}}}"
        return s

    @classmethod
    def load_from_file(cls, path):
        path = beam_path(path)
        s = path.read_text()
        s = cls.transform_to_json_like(s)
        parsed_config = json.loads(s)
        return cls(**parsed_config)

    def save_to_file(self, path):
        path = beam_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write(self._serialize_config(), ext='bin')

    def _serialize_config(self):
        config_str = f'name: "{self.name}"\n'
        config_str += f'platform: "{self.platform}"\n'
        for section_name in ['  input', '  output', 'instance_groups']:
            for section in getattr(self, section_name):
                config_str += f'{section_name} [\n'
                config_str += self._serialize_section(section)
                config_str += ']\n'
        return config_str

    @staticmethod
    def _serialize_section(section):
        return '\n'.join([f'  {key}: "{TritonConfig._serialize_value(key, value)}"' for key, value in section.items()])

    @staticmethod
    def _serialize_value(key, value):
        if key == 'dims':
            return '[' + ', '.join(map(str, value)) + ']'
        return value


class TritonClient(Processor):

    def __init__(self, scheme='http', host='localhost', port=8000, model_name=None, model_version=None,
                 verbose=False, concurrency=1, connection_timeout=60.0, network_timeout=60.,
                 max_greenlets=None, ssl_options=None, *args, ssl_context_factory=None,
                 insecure=False, config=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.scheme = scheme
        self.host = host
        self.port = port
        self.model_name = model_name
        self.model_version = model_version
        self.verbose = verbose
        self.concurrency = concurrency
        self.connection_timeout = connection_timeout
        self.network_timeout = network_timeout
        self.max_greenlets = max_greenlets
        self.ssl_options = ssl_options
        self.ssl_context_factory = ssl_context_factory
        self.insecure = insecure
        self.ssl = scheme == 'https' or scheme == 'grpcs'
        self.config = config

    @lazy_property
    def client(self):
        if 'http' in self.scheme:
            from tritonclient.http import InferenceServerClient
        elif 'grpc' in self.scheme:
            from tritonclient.grpc import InferenceServerClient
        else:
            raise ValueError(f"Invalid scheme: {self.scheme}")

        url = f'{self.host}:{self.port}'

        return InferenceServerClient(url, concurrency=self.concurrency, connection_timeout=self.connection_timeout,
                                     network_timeout=self.network_timeout, max_greenlets=self.max_greenlets,
                                     ssl_options=self.ssl_options, ssl_context_factory=self.ssl_context_factory,
                                     verbose=self.verbose, insecure=self.insecure, ssl=self.ssl)

    def __call__(self, *args, **kwargs):
        # Create inputs for the inference request
        inputs = [self.client.InferInput("data_0", input_data.shape, "FP32")]
        inputs[0].set_data_from_numpy(input_data)

        # Send the inference request
        response = self.client.infer(self.model_name, inputs,)

        # Process the response
        output_data = response.as_numpy("fc6_1")

        return output_data