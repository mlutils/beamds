import json
import re

from dataclasses import dataclass, field

from ..path import beam_path

pytorch_to_triton_dtype_dict = {
        'float32': 'TYPE_FP32',
        'float16': 'TYPE_FP16',
        'int32': 'TYPE_INT32',
        'int64': 'TYPE_INT64',
        'int16': 'TYPE_INT16',
        'int8': 'TYPE_INT8',
        'uint8': 'TYPE_UINT8',
        'bool': 'TYPE_BOOL',
        'float64': 'TYPE_FP64',
        'bfloat16': 'TYPE_BF16',
    }


def pytorch_to_triton_dtype(dtype):
    dtype = str(dtype)
    return pytorch_to_triton_dtype_dict[dtype]



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
