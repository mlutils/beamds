import json
import re

from dataclasses import dataclass, field

from ..core import Processor
from ..path import beam_path, local_copy


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

    @staticmethod
    def _parse_config(lines):
        config_data = {'input': [], 'output': [], 'instance_groups': []}
        current_section = None
        section_lines = []

        for line in lines:
            line = line.strip()
            if ':' in line and not line.endswith('['):
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip().strip('"')
                if key in ['name', 'platform']:
                    config_data[key] = value
                elif key == 'max_batch_size':
                    config_data[key] = int(value)
            elif line.endswith('['):
                current_section = line.split('[')[0].strip()
                section_lines = []
            elif line.endswith(']'):
                if current_section in config_data:
                    parsed_section = TritonConfig._parse_section(section_lines)
                    config_data[current_section].append(parsed_section)
                current_section = None
                section_lines = []
            elif current_section:
                section_lines.append(line)

        return config_data

    @staticmethod
    def _parse_section(lines):
        section = {}
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key, value = key.strip(), value.strip().strip('"')
                if key in ['dims', 'shape']:
                    value = tuple(map(int, value.strip('[]').split(',')))
                section[key] = value
        return section

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
    pass
