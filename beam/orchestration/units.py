from typing import Union

class K8SUnits:
    def __init__(self, value: Union[int, float, str, "K8SUnits"], resource_type="memory"):
        self.resource_type = resource_type
        self.value = self.parse_value(value)

    def parse_value(self, value: Union[int, float, str, "K8SUnits"]):
        if isinstance(value, K8SUnits):
            return value.value
        elif isinstance(value, str):
            return self.parse_str(value)
        elif isinstance(value, (int, float)):
            return self.parse_numeric(value)
        else:
            raise ValueError(f"Unsupported type for K8SUnits value: {type(value)}")

    def parse_str(self, value: str):
        if self.resource_type == "cpu":
            if value.endswith('m'):
                # e.g., "500m" -> 500 (millicores)
                return int(value.replace('m', ''))
            else:
                # e.g., "0.5" -> 500m, "1" -> 1000m (1 core)
                return int(float(value) * 1000)
        elif self.resource_type == "memory":
            if value.endswith('Gi'):
                # e.g., "1Gi" -> 1 GiB in bytes
                return int(float(value.replace('Gi', '')) * 1000 ** 3)
            elif value.endswith('Mi'):
                # e.g., "500Mi" -> 500 MiB in bytes
                return int(float(value.replace('Mi', '')) * 1000 ** 2)
            else:
                # Default: if float < 1, interpret as MiB; otherwise, as GiB
                return int(float(value) * (1000 ** 3 if float(value) >= 1 else 1000 ** 2))

    def parse_numeric(self, value: Union[int, float]):
        if self.resource_type == "cpu":
            # Convert CPU float to millicores (e.g., 0.5 -> 500m)
            return int(value * 1000) if isinstance(value, float) else value * 1000
        elif self.resource_type == "memory":
            # Convert integer to GiB and float to MiB
            return int(value * 1000 ** 3) if isinstance(value, int) else int(value * 1000 ** 2)
        else:
            raise ValueError(f"Unsupported resource type: {self.resource_type}")

    @property
    def as_str(self):
        if self.resource_type == "memory":
            if self.value >= 1000 ** 3:
                return f"{self.value / 1000 ** 3}Gi"
            elif self.value >= 1000 ** 2:
                return f"{self.value / 1000 ** 2}Mi"
            else:
                return f"{self.value}B"  # Default to bytes if very small
        elif self.resource_type == "cpu":
            if self.value % 1000 == 0:
                return str(self.value // 1000)  # Full cores
            else:
                return f"{self.value}m"  # Millicores
        else:
            return str(self.value)

    def __str__(self):
        return self.as_str

    def __int__(self):
        return self.value
