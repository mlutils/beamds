from typing import Union


class K8SUnits:
    # manage transformation from numbers to k8s units (Gi, m, etc.)
    def __init__(self, value):
        if isinstance(value, int):
            self.value = value
        elif isinstance(value, K8SUnits):
            self.value = value.value
        elif isinstance(value, str):
            self.value = self.parse_str(value)

    def parse_str(self, value: str):
        if 'Gi' in value:
            return int(value.replace('Gi', '')) * 1000 ** 3
        elif 'm' in value:
            return int(value.replace('m', '')) * 1000 ** 2
        else:
            return int(value)

    @property
    def as_str(self):
        if self.value // 1000 ** 3 == self.value / 1000 ** 3 and self.value >= 1000 ** 3:
            return f"{self.value / 1000 ** 3}Gi"
        elif self.value // 1000 ** 2 == self.value / 1000 ** 2 and self.value >= 1000 ** 2:
            return f"{self.value / 1000 ** 2}m"
        elif self.value is None:
            return None
        else:
            return str(self.value)

    @property
    def as_number(self):
        return self.value

    def __str__(self):
        return self.as_str

    def __int__(self):
        return self.value
