import time
import random
from faker import Faker
from ..processor import Processor


class BeamFakeAlg(Processor):
    def __init__(self, sleep_time=1, variance=0., error_rate=0., **faker_fields):
        """
        Initialize the BeamFakeAlg.

        :param sleep_time: Base sleep time in seconds.
        :param variance: Variance in sleep time in seconds.
        :param error_rate: Probability of raising an exception to simulate errors.
        :param faker_fields: Configurable fields for Faker to generate.
        """
        super().__init__()
        self.sleep_time = sleep_time
        self.variance = variance
        self.error_rate = error_rate
        faker_fields = {f: v for f, v in faker_fields.items() if not f.startswith('_')}
        self.faker_fields = faker_fields if len(faker_fields) > 0 else {'name': 'name', 'email': 'email',
                                                                        'address': 'address'}
        self.fake = Faker()

    def run(self, x):
        """
        Process the input data and return fake results.

        :param x: Input data.
        :return: Fake data results or raises an exception.
        """
        # Simulate processing time
        actual_sleep_time = self.sleep_time + random.uniform(-self.variance, self.variance)
        time.sleep(actual_sleep_time)

        # Simulate error
        if random.random() < self.error_rate:
            raise Exception("Simulated error")

        # Generate fake data based on input x (not used in this simple example)
        # You can modify this part to incorporate 'x' in the data generation logic
        data = {field: getattr(self.fake, field)() for field in self.faker_fields}

        return data