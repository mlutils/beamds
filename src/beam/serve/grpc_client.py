import grpc
import beam_grpc_pb2 as beamgrpc_pb2
import beam_grpc_pb2_grpc as beamgrpc_pb2_grpc
import pickle
from .beam_client import BeamClient  # Import your BeamClient
from functools import partial


class GRPCClient(BeamClient):

    def __init__(self, host, *args, **kwargs):
        super().__init__(host, *args, **kwargs)
        # Establish a gRPC channel
        self.channel = grpc.insecure_channel(host)
        # Create a stub (client proxy) for the BeamService
        self.stub = beamgrpc_pb2_grpc.BeamServiceStub(self.channel)

    def get_info(self):
        # Call the get_info RPC method
        response = self.stub.get_info(beamgrpc_pb2.info_request())
        return response.info_json

    def call_function(self, *args, **kwargs):
        # Serialize arguments and keyword arguments
        serialized_args = pickle.dumps(args)
        serialized_kwargs = pickle.dumps(kwargs)

        # Call the call_function RPC method
        response = self.stub.call_function(
            beamgrpc_pb2.func_request(args=serialized_args, kwargs=serialized_kwargs)
        )

        return pickle.loads(response.result)

    def query_algorithm(self, method_name, *args, **kwargs):
        # Serialize arguments and keyword arguments
        serialized_args = pickle.dumps(args)
        serialized_kwargs = pickle.dumps(kwargs)

        # Call the query_algorithm RPC method
        response = self.stub.query_algorithm(
            beamgrpc_pb2.method_request(method_name=method_name, args=serialized_args, kwargs=serialized_kwargs)
        )

        return pickle.loads(response.result)

    def set_variable(self, name, value):
        # Serialize the value
        serialized_value = pickle.dumps(value)

        # Call the set_variable RPC method
        response = self.stub.set_variable(
            beamgrpc_pb2.set_variable_request(name=name, value=serialized_value)
        )

        return response.success

    def get_variable(self, name):
        # Call the get_variable RPC method
        response = self.stub.get_variable(beamgrpc_pb2.get_variable_request(name=name))

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
