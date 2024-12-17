from typing import Union

class K8SUnits:
    MAX_MEMORY = 512 * 1024 ** 3  # 512Gi in bytes
    MAX_CPU = 24 * 1000  # 24 cores in millicores

    def __init__(self, value: Union[int, float, str, "K8SUnits"], resource_type="memory"):
        self.resource_type = resource_type
        self.value = self.parse_value(value)
        self.validate_limits()

    def parse_value(self, value: Union[int, float, str, "K8SUnits"]):
        if isinstance(value, K8SUnits):
            return value.value
        elif isinstance(value, str):
            return self.parse_str(value)
        elif isinstance(value, (float, int)):
            return self.parse_numeric(float(value))
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
                # Interpret decimal as a mix of GiB and MiB, e.g., 1.3 means 1 GiB + 300 MiB
                return self.parse_mixed_memory(float(value))
    @staticmethod
    def parse_mixed_memory(value: float):
        gi = int(value)  # Whole GiB
        mi_fraction = (value - gi) * 1024  # Remaining fraction as MiB
        return gi * 1024 ** 3 + int(mi_fraction) * 1024 ** 2

    def parse_numeric(self, value: Union[int, float]):
        if self.resource_type == "cpu":
            return int(value * 1000)
        elif self.resource_type == "memory":
            if isinstance(value, int):
                return value * 1024 ** 3  # Treat integers as GiB
            else:
                return self.parse_mixed_memory(value)  # Treat floats as mixed GiB + MiB
        else:
            raise ValueError(f"Unsupported resource type: {self.resource_type}")

    def validate_limits(self):
        if self.resource_type == "memory" and self.value > self.MAX_MEMORY:
            raise ValueError(f"Memory limit exceeded: {self.value / 1024 ** 3:.2f}Gi exceeds max {self.MAX_MEMORY / 1024 ** 3:.2f}Gi")
        if self.resource_type == "cpu" and self.value > self.MAX_CPU:
            raise ValueError(f"CPU limit exceeded: {self.value / 1000:.2f} cores exceed max {self.MAX_CPU / 1000:.2f} cores")

    @property
    def as_str(self):
        if self.resource_type == "memory":
            # If the value is not an exact GiB, display it in MiB for accuracy
            if self.value % 1024 ** 3 != 0:
                return f"{self.value / 1024 ** 2:.0f}Mi"
            else:
                return f"{self.value / 1024 ** 3:.0f}Gi"
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
