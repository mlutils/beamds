from collections import defaultdict
from functools import partial
from ..utils import cached_property

from ..processor import Processor
from ..utils import as_numpy, check_type, as_tensor
from ..type import Types


triton_to_numpy_dtype_dict = {
    'BOOL': 'bool',
    'INT8': 'int8',
    'INT16': 'int16',
    'INT32': 'int32',
    'INT64': 'int64',
    'UINT8': 'uint8',
    'UINT16': 'uint16',
    'UINT32': 'uint32',
    'UINT64': 'uint64',
    'FP16': 'float16',
    'FP32': 'float32',
    'FP64': 'float64',
    'BYTES': 'bytes',
    'STRING': 'string',
    'UNDEFINED': 'undefined',
}


class TritonClient(Processor):

    def __init__(self, scheme='http', host='localhost', port=8000, model_name=None, model_version=None,
                 verbose=False, concurrency=1, connection_timeout=60.0, network_timeout=60.,
                 max_greenlets=None, ssl_options=None, *args, ssl_context_factory=None,
                 insecure=False, config=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.scheme = scheme
        self.host = host
        self.port = port
        self.model_name = model_name
        self.model_version = model_version
        self.verbose = verbose
        self.concurrency = concurrency
        self.connection_timeout = connection_timeout
        self.network_timeout = network_timeout
        self.max_greenlets = max_greenlets
        self.ssl_options = ssl_options
        self.ssl_context_factory = ssl_context_factory
        self.insecure = insecure
        self.ssl = scheme == 'https' or scheme == 'grpcs'
        self.config = config
        self.metadata_cache = defaultdict(dict)

    def get_metadata(self, model_name=None, model_version=None):

        if model_name in self.metadata_cache and model_version in self.metadata_cache[model_name]:
            return self.metadata_cache[model_name][model_version]

        model_name = model_name or self.model_name
        model_version = model_version or self.model_version or ''
        metadata = self.client.get_model_metadata(model_name, model_version=model_version)

        self.metadata_cache[model_name][model_version] = metadata
        return metadata

    @cached_property
    def metadata(self):
        return self.get_metadata(self.model_name, self.model_version)

    @property
    def is_alive(self):
        return self.client.is_server_live()

    @cached_property
    def infer_input(self):
        if 'http' in self.scheme:
            from tritonclient.http import InferInput
        elif 'grpc' in self.scheme:
            from tritonclient.grpc import InferInput
        else:
            raise ValueError(f"Invalid scheme: {self.scheme}")
        return InferInput

    @cached_property
    def infer_requested_output(self):
        if 'http' in self.scheme:
            from tritonclient.http import InferRequestedOutput
        elif 'grpc' in self.scheme:
            from tritonclient.grpc import InferRequestedOutput
        else:
            raise ValueError(f"Invalid scheme: {self.scheme}")
        return InferRequestedOutput

    @cached_property
    def client(self):
        if 'http' in self.scheme:
            from tritonclient.http import InferenceServerClient
        elif 'grpc' in self.scheme:
            from tritonclient.grpc import InferenceServerClient
        else:
            raise ValueError(f"Invalid scheme: {self.scheme}")

        url = f'{self.host}:{self.port}'

        return InferenceServerClient(url, concurrency=self.concurrency, connection_timeout=self.connection_timeout,
                                     network_timeout=self.network_timeout, max_greenlets=self.max_greenlets,
                                     ssl_options=self.ssl_options, ssl_context_factory=self.ssl_context_factory,
                                     verbose=self.verbose, insecure=self.insecure, ssl=self.ssl)

    def call_model(self, *args, model_name=None, model_version=None):
        model_name = model_name or self.model_name
        model_version = model_version or self.model_version or ''
        inputs_metadata = []
        org_type = check_type(args[0])
        org_device = args[0].device if org_type.minor == Types.tensor else None
        inputs = []
        metadata = self.get_metadata(model_name, model_version)
        for i, input_metadata in enumerate(metadata['inputs']):
            # input_shape = [1] + input_shape
            input_metadata = self.infer_input(input_metadata['name'], input_metadata['shape'],
                                              input_metadata['datatype'])
            input = args[i]
            input = input_metadata.set_data_from_numpy(
                    as_numpy(input, dtype=triton_to_numpy_dtype_dict[input_metadata.datatype()]))
            inputs_metadata.append(input_metadata)
            inputs.append(input)

        outputs_metadata = []
        for output_metadata in metadata['outputs']:
            output_metadata = self.infer_requested_output(output_metadata['name'])
            outputs_metadata.append(output_metadata)

        response = self.client.infer(model_name=model_name, inputs=inputs, outputs=outputs_metadata)
        outputs = []
        for output_metadata in outputs_metadata:
            out = response.as_numpy(output_metadata.name())
            if org_type.minor == Types.tensor:
                out = as_tensor(out, device=org_device)
            outputs.append(out)
            
        if len(outputs) == 1:
            outputs = outputs[0]

        return outputs

    def __call__(self, *args):
        return self.call_model(*args)

    def getattr(self, item):

        if self.model_name is None:
            func = partial(self.call_model, model_name=item)
        else:
            func = partial(self.call_model, model_name=self.model_name, model_version=item)
        return func

    def __getitem__(self, item):
        parts = item.split('/')
        model_name = parts[0]
        model_version = parts[1] if len(parts) > 1 else None
        func = partial(self.call_model, model_name=model_name, model_version=model_version)
        return func
