import io
import json

import grpc
from concurrent import futures
from .beam_grpc_pb2 import (SetVariableRequest, SetVariableResponse, GetVariableRequest,
                            GetVariableResponse, QueryAlgorithmRequest, QueryAlgorithmResponse, GetInfoResponse)
from .beam_grpc_pb2_grpc import add_BeamServiceServicer_to_server, BeamServiceServicer
from .server import BeamServer


class GRPCServer(BeamServer, BeamServiceServicer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        add_BeamServiceServicer_to_server(self, self.grpc_server)

    def SetVariable(self, request, context):
        # Directly use self to handle the logic
        try:
            value = io.BytesIO(request.value)
            success = self.set_variable(client=request.client, name=request.name, value=value)
            return SetVariableResponse(success=success)
        except Exception as e:
            context.abort(grpc.StatusCode.INTERNAL, str(e))

    def GetVariable(self, request, context):
        # Directly use self to handle the logic
        try:
            value = self.get_variable(client=request.client, name=request.name)
            return GetVariableResponse(value=value.getvalue())
        except AttributeError:
            context.abort(grpc.StatusCode.NOT_FOUND, f"Variable {request.name} not found")
        except Exception as e:
            context.abort(grpc.StatusCode.INTERNAL, str(e))

    def QueryAlgorithm(self, request, context):
        # Directly use self to handle the logic
        try:
            args = io.BytesIO(request.args) if request.args else None
            kwargs = io.BytesIO(request.kwargs) if request.kwargs else None
            results = self.query_algorithm(client=request.client, method=request.method, args=args, kwargs=kwargs)
            return QueryAlgorithmResponse(results=results.getvalue())
        except Exception as e:
            context.abort(grpc.StatusCode.INTERNAL, str(e))

    def GetInfo(self, request, context):
        try:
            info_data = self.get_info()  # Assuming `get_info` is implemented in `BeamServer`
            # Ensure info_data is serialized properly if it's not a simple string
            info_data = json.dumps(info_data)
            return GetInfoResponse(info=info_data)
        except Exception as e:
            context.abort(grpc.StatusCode.INTERNAL, str(e))

    def _run(self, host="0.0.0.0", port=None, **kwargs):
        address = f'{host}:{port}'
        self.grpc_server.add_insecure_port(address)
        print(f"Starting gRPC server on {address}")
        self.grpc_server.start()
        self.grpc_server.wait_for_termination()