import threading

from .dispatcher import BeamDispatcher
from ..core import Processor
from ..serve.http_server import HTTPServer
from ..serve.server import BeamServer
from ..logger import beam_logger as logger
import websockets
import asyncio
from celery.signals import task_postrun
from flask import Flask, request, jsonify, send_file
from ..utils import ThreadSafeDict


class BeamAsyncServer(HTTPServer):

    def __init__(self, *routes, name=None, broker=None, backend=None,
                 broker_username=None, broker_password=None, broker_port=None, broker_scheme=None, broker_host=None,
                 backend_username=None, backend_password=None, backend_port=None, backend_scheme=None,
                 backend_host=None, use_torch=True, batch=None, max_wait_time=1.0, max_batch_size=10,
                 tls=False, n_threads=4, application=None, postrun=None, **kwargs):

        predefined_attributes = {k: 'method' for k in routes}

        self.dispatcher = BeamDispatcher(name=name, broker=broker, backend=backend,
                                         broker_username=broker_username, broker_password=broker_password,
                                         broker_port=broker_port, broker_scheme=broker_scheme, broker_host=broker_host,
                                         backend_username=backend_username, backend_password=backend_password,
                                         backend_port=backend_port, backend_scheme=backend_scheme,
                                         backend_host=backend_host, serve='global', **kwargs)

        super().__init__(obj=self.dispatcher, use_torch=use_torch, batch=batch,
                        max_wait_time=max_wait_time, max_batch_size=max_batch_size,
                        tls=tls, n_threads=n_threads, application=application,
                        predefined_attributes=predefined_attributes, **kwargs)

        self.tasks = ThreadSafeDict()

        if postrun is None:
            self.postrun_callback = self.postrun
        else:
            self.postrun_callback = postrun

    def query_algorithm(self, client, method, *args, **kwargs):

        if client == 'beam':
            postrun_args = request.files['postrun_args']
            postrun_kwargs = request.files['postrun_kwargs']
            args = request.files['args']
            kwargs = request.files['kwargs']
            ws_client_id = request.files['ws_client_id']

            postrun_args = self.load_function(postrun_args)
            postrun_kwargs = self.load_function(postrun_kwargs)
            ws_client_id = self.load_function(ws_client_id)

        else:
            data = request.get_json()
            postrun_args = data.pop('postrun_args', [])
            postrun_kwargs = data.pop('postrun_kwargs', {})
            ws_client_id = data.pop('ws_client_id', None)

            args = data.pop('args', [])
            kwargs = data.pop('kwargs', {})

        task_id = BeamServer.query_algorithm(client, method, args, kwargs)

        metadata = self.request_metadata(client=client, method=method)
        self.tasks[task_id] = {'metadata': metadata, 'postrun_args': postrun_args,
                               'postrun_kwargs': postrun_kwargs, 'ws_client_id': ws_client_id}

        if client == 'beam':
            return send_file(task_id, mimetype="text/plain")
        else:
            return jsonify(task_id)

    @task_postrun.connect
    def postprocess(self, task_id, task, args, kwargs, retval, state):
        logger.info(f"Task {task_id} finished with state {state}.")
        task_inf = self.tasks[task_id]

        # Send notification to the client via WebSocket
        if task_inf['ws_client_id'] is not None:
            websocket = self.websocket_clients.get(str(task_inf['ws_client_id']))
            if websocket and websocket.open:
                asyncio.run(websocket.send({"task_id": task_id, "state": state}))
                asyncio.run(websocket.close())
                del self.websocket_clients[str(task_inf['ws_client_id'])]

        self.postrun_callback(task_args=args, task_kwargs=kwargs, retval=retval, state=state, task=task, **task_inf)

    def postrun(self, task_args=None, task_kwargs=None, retval=None, state=None, task=None, metadata=None,
                postrun_args=None, postrun_kwargs=None, **kwargs):
        pass

    def run(self, *args, **kwargs):

        # start the web socket server
        super().run(*args, **kwargs)
