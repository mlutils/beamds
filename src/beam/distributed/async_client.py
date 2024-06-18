import io
import json
from threading import Thread
from uuid import uuid4 as uuid
import websocket
from ..logging import beam_logger as logger

from ..path import normalize_host
from ..serve.http_client import HTTPClient


class AsyncClient(HTTPClient):

    def __init__(self, hostname=None, ws_port=None, ws_tls=False, postrun=None, enable_websocket=True,  **kwargs):
        super().__init__(hostname=hostname, **kwargs)

        metadata = self.info.get('metadata', {})
        if ws_port is None and 'ws_port' in metadata:
            ws_port = metadata['ws_port']
            self.ws_application = metadata['ws_application']
        else:
            self.ws_application = 'ws' if not ws_tls else 'wss'

        self.ws_host = normalize_host(hostname, ws_port)
        if postrun is None:
            self.postrun_callback = self.postrun
        else:
            self.postrun_callback = postrun

        self.client_id = f"ws-client-{uuid()}"

        if enable_websocket:
            self.init_websocket()

        self.ws = None
        self.wst = None

    def set_callback(self, callback):
        self.postrun_callback = callback

    def post(self, path, *args, postrun_args=None, postrun_kwargs=None, **kwargs):

        io_args = io.BytesIO()
        self.dump_function(args, io_args)
        io_args.seek(0)

        io_kwargs = io.BytesIO()
        self.dump_function(kwargs, io_kwargs)
        io_kwargs.seek(0)

        io_postrun_args = io.BytesIO()
        self.dump_function(postrun_args, io_postrun_args)
        io_postrun_args.seek(0)

        io_postrun_kwargs = io.BytesIO()
        self.dump_function(postrun_kwargs, io_postrun_kwargs)
        io_postrun_kwargs.seek(0)

        io_ws_client_id = io.BytesIO()
        self.dump_function(self.client_id, io_ws_client_id)
        io_ws_client_id.seek(0)

        response = self._post(path, io_args, io_kwargs, postrun_args=io_postrun_args,
                              postrun_kwargs=io_postrun_kwargs, ws_client_id=io_ws_client_id)

        return response

    def init_websocket(self):
        self.ws = websocket.WebSocketApp(f"{self.ws_application}://{self.ws_host}/",
                                         on_open=self.on_open,
                                         on_message=self.on_message,
                                         on_error=self.on_error,
                                         on_close=self.on_close)
        # Run the WebSocket client in a separate thread
        self.wst = Thread(target=self.ws.run_forever)
        self.wst.start()

    def postrun(self, result):
        pass

    def poll(self, task_id, timeout=None):
        params = {'task_id': task_id}
        if timeout is not None:
            params['timeout'] = timeout
        return self.get('/poll/beam', params=params)

    def on_message(self, ws, message):
        data = json.loads(message)

        # Extract task_id and state from the message
        task_id = data.get('task_id')
        state = data.get('state')
        metadata = data.get('metadata', None)

        if state == 'SUCCESS':
            result = self.poll(task_id)
            self.postrun_callback(result)
        else:
            logger.error(f"Task {task_id} failed with state {state} and metadata: {metadata}")

    def on_error(self, ws, error):
        logger.error(f"WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        logger.warning(f"WebSocket closed with status code {close_status_code} and message: {close_msg}")
        self.init_websocket()

    def on_open(self, ws):
        logger.info(f"Opening websocket at {self.ws_application}://{self.ws_host}/: client_id: {self.client_id}")
        ws.send(self.client_id)
