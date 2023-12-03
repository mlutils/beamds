import json
import grpc
from concurrent import futures
import beam_grpc_pb2 as beamgrpc_pb2
from .beam_grpc_pb2 import pickled_response, info_response, set_variable_response, get_variable_response
import beam_grpc_pb2_grpc as beamgrpc_pb2_grpc
import pickle
from .beam_server import BeamServer  # Import your BeamServer


class GRPCServer(BeamServer, beamgrpc_pb2_grpc.BeamServiceServicer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, application='grpc', **kwargs)

    def query_algorithm(self, request, context):
        method_name = request.method_name
        args = pickle.loads(request.args)
        kwargs = pickle.loads(request.kwargs)
        # Assuming 'query_algorithm' method in BeamServer class
        result = super().query_algorithm(method_name, args, kwargs)
        return beamgrpc_pb2.pickled_response(result=pickle.dumps(result))

    def call_function(self, request, context):
        args = pickle.loads(request.args)
        kwargs = pickle.loads(request.kwargs)
        # Assuming 'call_function' method in BeamServer class
        result = super().call_function(args, kwargs)
        return beamgrpc_pb2.pickled_response(result=pickle.dumps(result))

    def get_info(self, request, context):
        # Assuming 'get_info' method in BeamServer class returns a dictionary
        info = super().get_info()
        return beamgrpc_pb2.info_response(info_json=json.dumps(info))

    def set_variable(self, request, context):
        name = request.name
        value = pickle.loads(request.value)
        success = super().set_variable(name, value)  # Assuming this method exists and returns boolean
        return beamgrpc_pb2.set_variable_response(success=success)

    def get_variable(self, request, context):
        name = request.name
        value = super().get_variable(name)  # Assuming this method exists and returns the value
        return beamgrpc_pb2.get_variable_response(value=pickle.dumps(value))

    def _run(self, host="0.0.0.0", port=None, **kwargs):
        self.grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=self.n_threads))
        beamgrpc_pb2_grpc.add_BeamServiceServicer_to_server(self, self.grpc_server)
        self.grpc_server.add_insecure_port(f"{host}:{port}")
        self.grpc_server.start()
        self.grpc_server.wait_for_termination()

