import grpc
from .beam_grpc_pb2 import (pickled_response, info_response, set_variable_response,
                            get_variable_response, method_request, info_request, set_variable_request,
                            get_variable_request, func_request)
from .beam_grpc_pb2_grpc import BeamServiceStub
import pickle
from .client import BeamClient  # Import your BeamClient
from functools import partial


class GRPCClient(BeamClient):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Establish a gRPC channel
        self.channel = grpc.insecure_channel(self.host)
        # Create a stub (client proxy) for the BeamService
        self.stub = BeamServiceStub(self.channel)

    def get_info(self):
        # Call the get_info RPC method
        response = self.stub.get_info(info_request())
        return response.info_json

    def call_function(self, *args, **kwargs):
        # Serialize arguments and keyword arguments
        serialized_args = pickle.dumps(args)
        serialized_kwargs = pickle.dumps(kwargs)

        # Call the call_function RPC method
        response = self.stub.call_function(
            func_request(args=serialized_args, kwargs=serialized_kwargs)
        )

        return pickle.loads(response.result)

    def query_algorithm(self, method_name, *args, **kwargs):
        # Serialize arguments and keyword arguments
        serialized_args = pickle.dumps(args)
        serialized_kwargs = pickle.dumps(kwargs)

        # Call the query_algorithm RPC method
        response = self.stub.query_algorithm(
            method_request(method_name=method_name, args=serialized_args, kwargs=serialized_kwargs)
        )

        return pickle.loads(response.result)

    def set_variable(self, name, value):
        # Serialize the value
        serialized_value = pickle.dumps(value)

        # Call the set_variable RPC method
        response = self.stub.set_variable(
            set_variable_request(name=name, value=serialized_value)
        )

        return response.success

    def get_variable(self, name):
        # Call the get_variable RPC method
        response = self.stub.get_variable(get_variable_request(name=name))

        return pickle.loads(response.value)

    def __getattr__(self, item):
        if item.startswith('_'):
            return super(GRPCClient, self).__getattr__(item)

        if item not in self.attributes:
            self.clear_cache('info')

        attribute_type = self.attributes[item]
        if attribute_type == 'variable':
            return self.get_variable(item)
        elif attribute_type == 'method':
            return partial(self.query_algorithm, item)
        raise ValueError(f"Unknown attribute type: {attribute_type}")

    def __setattr__(self, key, value):
        if key in ['host', '_info', '_lazy_cache', 'channel', 'stub']:
            super(GRPCClient, self).__setattr__(key, value)
        else:
            self.set_variable(key, value)
