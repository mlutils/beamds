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
    def __init__(self, url=None, scheme='http', host='localhost', port=8000, model_name=None, model_version=None,
                 verbose=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        path = None
        if model_name is not None:
            model_version = model_version or ''
            path = f"{model_name}/{model_version}"

        self.url = BeamURL(url=url, scheme=scheme, host=host, port=port, path=path)

        self.model_name = self.url.path.split('/')[0]
        self.model_version = self.url.path.split('/')[1] if len(self.url.path.split('/')) > 1 else ''
        self.host = f"{self.url.hostname}:{self.url.port}"

    @lazy_property
    def client(self):
        if self.url.scheme == 'http':
            from tritonclient.http import InferenceServerClient
        elif self.url.scheme == 'grpc':
            from tritonclient.grpc import InferenceServerClient
        else:
            raise ValueError(f"Invalid scheme: {self.url.scheme}")

        return InferenceServerClient(url=self.host, verbose=self.verbose,
                                     model_version=self.model_version)

    def __call__(self, *args, **kwargs):
        # Create inputs for the inference request
        inputs = [self.client.InferInput("data_0", input_data.shape, "FP32")]
        inputs[0].set_data_from_numpy(input_data)

        # Send the inference request
        response = self.client.infer(self.model_name, inputs,)

        # Process the response
        output_data = response.as_numpy("fc6_1")

        return output_data