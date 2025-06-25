import json
import time
from threading import Thread
import io
import websockets
import asyncio
from flask import request, jsonify, send_file

from ..serve.http_server import HTTPServer
from ..serve.server import BeamServer
from ..logging import beam_logger as logger
from ..utils import ThreadSafeDict, find_port


class AsyncServer(HTTPServer):

    def __init__(self, dispatcher, *args, asynchronous=True, postrun=None, ws_tls=False, **kwargs):

        super().__init__(dispatcher, *args, **kwargs)

        self.app.add_url_rule('/poll/<client>/', view_func=self.poll)

        self.tasks = ThreadSafeDict()
        self.ws_clients = ThreadSafeDict()
        # task_postrun.connect(self.postprocess)
        self.asynchronous = asynchronous

        if ws_tls:
            import ssl
            # Create an SSL context for wss
            self.ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            # self.ssl_context.load_cert_chain(certfile='path/to/cert.pem', keyfile='path/to/key.pem')
        else:
            self.ssl_context = None

        self.ws_application = 'ws' if not ws_tls else 'wss'
        self.ws_port = None

        if postrun is None:
            self.postrun_callback = self.postrun
        else:
            self.postrun_callback = postrun

    def run_worker(self):
        pass

    def poll(self, client):

        task_id = request.args.get('task_id')
        timeout = request.args.get('timeout', 0)
        result = self.obj.poll(task_id, timeout=timeout)

        if client == 'beam':
            io_results = io.BytesIO()
            self.dump_function(result, io_results)
            io_results.seek(0)
            return send_file(io_results, mimetype="text/plain")
        else:
            return jsonify(result)

    async def websocket_handler(self, ws):
        # Wait for the client to send its client_id

        client_id = await ws.recv()
        logger.info(f"New WebSocket client connected: {client_id}")
        self.ws_clients[client_id] = ws
        await ws.wait_closed()
        # self.ws_clients.pop(client_id)

    async def run_ws_server(self, host, port):
        logger.info("Starting WebSocket server...")
        async with websockets.serve(self.websocket_handler, host, port, ssl=self.ssl_context):
            await asyncio.Future()  # Run forever until Future is cancelled

    def run_non_blocking(self, **kwargs):
        self.run(non_blocking=True, **kwargs)

    def run(self, ws_host=None, ws_port=None, enable_websocket=True, non_blocking=False, **kwargs):

        self.run_worker()

        if enable_websocket and self.asynchronous:
            if ws_host is None:
                ws_host = "0.0.0.0"
            ws_port = find_port(port=ws_port, get_port_from_beam_port_range=True, application=self.ws_application)
            self.ws_port = ws_port
            logger.info(f"Opening a Websocket ({self.ws_application}) serve on port: {ws_port}")

            # Run the WebSocket server as an asyncio task
            # Thread(target=self.run_ws_server, args=(ws_host, ws_port)).start()
            Thread(target=lambda: asyncio.run(self.run_ws_server(ws_host, ws_port))).start()

        super().run(non_blocking=True, **kwargs)

        if not non_blocking:
            self.routine()
        else:
            Thread(target=self.routine).start()

    def routine(self):

        while True:
            try:
                sleep = True
                for task_id, task in self.tasks.items():
                    res = task['async_result']
                    if res.is_ready:
                        sleep = False
                        task_inf = self.tasks.pop(task_id)
                        self.postprocess(task_id=task_id, async_result=res, task_inf=task_inf)
                if sleep:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt, exiting...")
                break

    @property
    def real_object(self):
        return self.obj.obj

    @property
    def metadata(self):
        return {'ws_application': self.ws_application, 'ws_port': self.ws_port}

    def query_algorithm(self, client, method, *args, **kwargs):

        if not self.asynchronous:
            return super().query_algorithm(client, method, *args, **kwargs)

        if client == 'beam':
            postrun_args = request.files['postrun_args']
            postrun_kwargs = request.files['postrun_kwargs']
            args = request.files['args']
            kwargs = request.files['kwargs']
            ws_client_id = request.files['ws_client_id']

            postrun_args = self.load_function(postrun_args, **self.lf_kwargs)
            postrun_kwargs = self.load_function(postrun_kwargs, **self.lf_kwargs)
            ws_client_id = self.load_function(ws_client_id, **self.lf_kwargs)

        else:
            data = request.get_json()
            postrun_args = data.pop('postrun_args', [])
            postrun_kwargs = data.pop('postrun_kwargs', {})
            ws_client_id = data.pop('ws_client_id', None)

            args = data.pop('args', [])
            kwargs = data.pop('kwargs', {})

        if method not in ['poll']:
            async_result = BeamServer.query_algorithm(self, client, method, args, kwargs, return_raw_results=True)
            task_id = async_result.hex
            metadata = self.request_metadata(client=client, method=method)
            self.tasks[task_id] = {'metadata': metadata, 'postrun_args': postrun_args,
                                            'postrun_kwargs': postrun_kwargs, 'ws_client_id': ws_client_id,
                                            'async_result': async_result}

            if client == 'beam':
                io_results = io.BytesIO()
                self.dump_function(task_id, io_results)
                io_results.seek(0)
                return send_file(io_results, mimetype="text/plain")
            else:
                return jsonify(task_id)

        else:
            return BeamServer.query_algorithm(self, client, method, args, kwargs)

    def postprocess(self, task_id=None, task_inf=None, async_result=None):

        state = async_result.state
        args = async_result.args
        kwargs = async_result.kwargs
        retval = async_result.value

        logger.info(f"Task {task_id} finished with state {state}.")

        if task_inf is None:
            logger.warning(f"Task {task_id} not found in tasks dict")
            return

        # Send notification to the client via WebSocket
        if task_inf['ws_client_id'] is not None:
            client_id = task_inf['ws_client_id']
            ws = self.ws_clients.get(client_id)
            if ws and ws.open:
                asyncio.run(ws.send(json.dumps({"task_id": task_id, "state": state})))
            else:
                if ws:
                    asyncio.run(ws.close())
                    self.ws_clients.pop(client_id)

        self.postrun_callback(task_args=args, task_kwargs=kwargs, retval=retval, state=state, task=async_result, **task_inf)

    def postrun(self, task_args=None, task_kwargs=None, retval=None, state=None, task=None, metadata=None,
                postrun_args=None, postrun_kwargs=None, **kwargs):
        pass


class AsyncCeleryServer(AsyncServer):

    def __init__(self, obj, routes=None, name=None, asynchronous=True, broker=None, backend=None,
                 broker_username=None, broker_password=None, broker_port=None, broker_scheme=None, broker_host=None,
                 backend_username=None, backend_password=None, backend_port=None, backend_scheme=None,
                 backend_host=None, use_torch=True, batch=None, max_wait_time=1.0, max_batch_size=10,
                 tls=False, n_threads=4, application=None, n_workers=1, broker_log_level='INFO', **kwargs):

        from .celery_dispatcher import CeleryDispatcher
        from .celery_worker import CeleryWorker

        if routes is None:
            routes = []
        self.worker = CeleryWorker(obj, *routes, name=name, n_workers=n_workers, daemon=True,
                                   broker=broker, backend=backend,
                                   broker_username=broker_username, broker_password=broker_password,
                                   broker_port=broker_port,
                                   broker_scheme=broker_scheme, broker_host=broker_host,
                                   backend_username=backend_username, backend_password=backend_password,
                                   backend_port=backend_port,
                                   backend_scheme=backend_scheme,
                                   backend_host=backend_host, log_level=broker_log_level)

        predefined_attributes = {k: 'method' for k in self.worker.routes}
        dispatcher = CeleryDispatcher(name=self.worker.name, broker=broker, backend=backend,
                                           broker_username=broker_username, broker_password=broker_password,
                                           broker_port=broker_port, broker_scheme=broker_scheme,
                                           broker_host=broker_host,
                                           backend_username=backend_username, backend_password=backend_password,
                                           backend_port=backend_port, backend_scheme=backend_scheme,
                                           backend_host=backend_host, log_level=broker_log_level,
                                           asynchronous=asynchronous)

        application = application or 'distributed_celery_async'
        super().__init__(dispatcher, name=name, asynchronous=asynchronous, use_torch=use_torch, batch=batch,
                         max_wait_time=max_wait_time, max_batch_size=max_batch_size,
                         tls=tls, n_threads=n_threads, application=application,
                         predefined_attributes=predefined_attributes, **kwargs)

    def run_worker(self):
        self.worker.run()


class AsyncRayServer(AsyncServer):

    def __init__(self, obj, routes=None, name=None, asynchronous=True, address=None, port=None, use_torch=True, batch=None,
                 max_wait_time=1.0, max_batch_size=10, tls=False, n_threads=4, application=None, ray_kwargs=None,
                 **kwargs):

        from .ray_dispatcher import RayDispatcher

        if routes is None:
            routes = []

        predefined_attributes = {k: 'method' for k in routes}
        dispatcher = RayDispatcher(obj, *routes, name=name, address=address, port=port, ray_kwargs=ray_kwargs,
                                   asynchronous=asynchronous, **kwargs)

        application = application or 'distributed_ray_async'
        super().__init__(dispatcher, name=name, asynchronous=asynchronous, use_torch=use_torch, batch=batch,
                         max_wait_time=max_wait_time, max_batch_size=max_batch_size,
                         tls=tls, n_threads=n_threads, application=application,
                         predefined_attributes=predefined_attributes, **kwargs)
