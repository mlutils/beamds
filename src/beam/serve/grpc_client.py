import grpc
from .beam_grpc_pb2 import SetVariableRequest, GetVariableRequest, QueryAlgorithmRequest
from .beam_grpc_pb2_grpc import BeamServiceStub


class GRPCClient(BeamClient):
    def __init__(self, *args, host="localhost", port=50051, **kwargs):
        super().__init__(*args, **kwargs)
        # Establishing the channel
        self.channel = grpc.insecure_channel(f'{host}:{port}')
        # Creating a stub (client)
        self.stub = BeamServiceStub(self.channel)

    def set_variable(self, name, value, client='beam'):
        """
        Set a variable on the server.

        :param name: The name of the variable to set.
        :param value: The value to set the variable to.
        :param client: The client identifier.
        """
        request = SetVariableRequest(client=client, name=name, value=value)
        return self.stub.SetVariable(request)

    def get_variable(self, name, client='beam'):
        """
        Get a variable's value from the server.

        :param name: The name of the variable to get.
        :param client: The client identifier.
        """
        request = GetVariableRequest(client=client, name=name)
        response = self.stub.GetVariable(request)
        return response.value

    def query_algorithm(self, method, args=None, kwargs=None, client='beam'):
        """
        Query an algorithm on the server.

        :param method: The method name of the algorithm to query.
        :param args: The arguments to pass to the algorithm.
        :param kwargs: The keyword arguments to pass to the algorithm.
        :param client: The client identifier.
        """
        if args is None:
            args = b''
        if kwargs is None:
            kwargs = b''
        request = QueryAlgorithmRequest(client=client, method=method, args=args, kwargs=kwargs)
        response = self.stub.QueryAlgorithm(request)
        return response.results

    def close(self):
        """Close the gRPC channel."""
        self.channel.close()
