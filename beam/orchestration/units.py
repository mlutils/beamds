from typing import Union


class K8SUnits:
    def __init__(self, value: Union[int, str, "K8SUnits"], resource_type="memory"):
        self.resource_type = resource_type
        self.value = self.parse_value(value)

    def parse_value(self, value: Union[int, str, "K8SUnits"]):
        if isinstance(value, K8SUnits):
            return value.value
        elif isinstance(value, str):
            return self.parse_str(value)
        elif isinstance(value, int):
            # If it's an integer and we're dealing with memory, interpret as GiB
            if self.resource_type == "memory":
                return value * 1000**3
            else:
                # For CPU, just use the integer value as it is (in cores)
                return value
        else:
            raise ValueError(f"Unsupported type for K8SUnits value: {type(value)}")

    def parse_str(self, value: str):
        if value.endswith('Gi'):
            return int(float(value.replace('Gi', ''))) * 1000**3
        elif value.endswith('Mi'):
            return int(float(value.replace('Mi', ''))) * 1000**2
        elif value.endswith('m'):
            # For CPU units, where 'm' means milli (e.g., 100m CPU means 100 milliCPUs)
            return int(value.replace('m', ''))
        else:
            # If no unit is specified, interpret as GiB for memory or cores for CPU
            return int(value) * (1000**3 if self.resource_type == "memory" else 1)

    @property
    def as_str(self):
        if self.resource_type == "memory":
            if self.value >= 1000 ** 3:
                return f"{self.value / 1000 ** 3}Gi"
            elif self.value >= 1000 ** 2:
                return f"{self.value / 1000 ** 2}Mi"
            else:
                return f"{self.value}Mi"  # Default to Mi if very small
        elif self.resource_type == "cpu":
            if self.value % 1000 == 0:
                # If the value is a multiple of 1000, express it in cores
                return str(self.value // 1000)
            else:
                # Otherwise, express in millicores
                return f"{self.value * 1000}m"
        else:
            # Fallback for any other types
            return str(self.value)

    def __str__(self):
        return self.as_str

    def __int__(self):
        return self.value
