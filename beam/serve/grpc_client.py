import io
import json

import grpc
from .beam_grpc_pb2 import SetVariableRequest, GetVariableRequest, QueryAlgorithmRequest, GetInfoRequest
from .beam_grpc_pb2_grpc import BeamServiceStub
from .client import BeamClient


class GRPCClient(BeamClient):
    def __init__(self, *args, host="localhost", port=50051, **kwargs):
        # Establishing the channel
        self.channel = grpc.insecure_channel(f'{host}:{port}')
        # Creating a stub (client)
        self.stub = BeamServiceStub(self.channel)
        super().__init__(*args, scheme='beam-grpc', **kwargs)

    def set_variable(self, name, value, client='beam'):
        """
        Set a variable on the server.

        :param name: The name of the variable to set.
        :param value: The value to set the variable to.
        :param client: The client identifier.
        """
        request = SetVariableRequest(client=client, name=name, value=value.getvalue())
        return self.stub.SetVariable(request)

    def get_variable(self, name, client='beam'):
        """
        Get a variable's value from the server.

        :param name: The name of the variable to get.
        :param client: The client identifier.
        """
        request = GetVariableRequest(client=client, name=name)
        response = self.stub.GetVariable(request)

        response = self.load_function(io.BytesIO(response.value), **self.lf_kwargs)

        return response

    def _post(self, method, io_args, io_kwargs, client='beam'):
        """
        Query an algorithm on the server.

        :param method: The method name of the algorithm to query.
        :param args: The arguments to pass to the algorithm.
        :param kwargs: The keyword arguments to pass to the algorithm.
        :param client: The client identifier.
        """

        io_args = b'' if io_args is None else io_args.getvalue()
        io_kwargs = b'' if io_kwargs is None else io_kwargs.getvalue()
        method = method.split('/')[-1]
        request = QueryAlgorithmRequest(client=client, method=method, args=io_args, kwargs=io_kwargs)
        response = self.stub.QueryAlgorithm(request)

        response = self.load_function(io.BytesIO(response.results), **self.lf_kwargs)
        return response

    def get_info(self):
        request = GetInfoRequest()  # If GetInfoRequest has fields, populate them here
        response = self.stub.GetInfo(request)
        info = json.loads(response.info)  # Assuming the response includes information as a string or serialized data
        return info

    def close(self):
        """Close the gRPC channel."""
        self.channel.close()
