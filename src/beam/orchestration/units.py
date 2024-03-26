from typing import Union


#
class K8SUnits:
    def __init__(self, value: Union[int, str, "K8SUnits"]):
        self.value = self.parse_value(value)

    def parse_value(self, value: Union[int, str, "K8SUnits"]):
        if isinstance(value, int):
            return value
        elif isinstance(value, K8SUnits):
            return value.value
        elif isinstance(value, str):
            return self.parse_str(value)
        else:
            raise ValueError(f"Unsupported type for K8SUnits value: {type(value)}")

    @staticmethod
    def parse_str(value: str):
        if value.endswith('Gi'):
            return int(float(value.replace('Gi', ''))) * 1000**3
        elif value.endswith('Mi'):
            return int(float(value.replace('Mi', ''))) * 1000**2
        elif value.endswith('m'):
            return int(float(value.replace('m', '')))
        else:
            return int(value)

    @property
    def as_str(self):
        if self.value >= 1000**3:
            return f"{self.value / 1000**3}Gi"
        elif self.value >= 1000**2:
            return f"{self.value / 1000**2}Mi"
        else:
            return f"{self.value}m"  # Assume milli-units by default for small numbers

    def __str__(self):
        return self.as_str

    def __int__(self):
        return self.value

