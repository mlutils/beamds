from typing import Iterator, Tuple, Optional, Union

import torch
import torch._dynamo as dynamo
from torch import nn, Tensor, device
from torch.nn.parallel import DistributedDataParallel as DDP

from ..path import beam_path, local_copy
from ..processor import Processor
from ..utils import recursive_clone, to_device, recursive_device


class BeamNN(nn.Module, Processor):
    """
    BeamNN is a wrapper class around PyTorch's nn.Module, with added functionalities
    from the Processor class. It allows dynamic integration of an existing nn.Module
    instance with additional processing capabilities.

    Attributes:
        _sample_input: A sample input for JIT tracing or other optimization methods.
        _module: The wrapped nn.Module instance.
    """

    def __init__(self, *args, _module=None, _model_type=None, **kwargs):

        """
        Initialize the BeamNN wrapper.

        Args:
            _module (nn.Module): The PyTorch module to be wrapped.
            *args, **kwargs: Additional arguments for the Processor initialization.
        """

        nn.Module.__init__(self)
        Processor.__init__(self, *args, **kwargs)
        self._sample_input = None
        self._sample_output = None
        self._module = _module
        self._model_type = _model_type or 'torch'

    @classmethod
    def from_module(cls, module, *args, hparams=None, **kwargs):

        """
        Class method to create a BeamNN object from an existing nn.Module.

        Args:
            module (nn.Module): The PyTorch module to be wrapped.

        Returns:
            BeamNN: A new BeamNN instance wrapping the provided module.
        """

        if isinstance(module, cls):
            if hparams is not None:
                module.update_hparams(hparams)
            beam_module = module
        else:
            beam_module = cls(*args, _module=module, hparams=hparams, **kwargs)
        return beam_module

    def __dir__(self):
        d = super().__dir__()
        if self.module_exists:
            d.extend(self._module.__dir__())
        return d

    def _mixin_method(self, method_name, *args, **kwargs):
        if self.module_exists:
            return getattr(self._module, method_name)(*args, **kwargs)
        return getattr(nn.Module, method_name)(self, *args, **kwargs)

    def parameters(self, *args, **kwargs):
        return self._mixin_method('parameters', *args, **kwargs)

    def named_parameters(self, *args, **kwargs):
        return self._mixin_method('named_parameters', *args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self._mixin_method('state_dict', *args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        return self._mixin_method('load_state_dict', state_dict, *args, **kwargs)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        return self._mixin_method('_load_from_state_dict', state_dict, prefix, local_metadata, strict,
                                  missing_keys, unexpected_keys, error_msgs)

    def apply(self, fn):
        return self._mixin_method('apply', fn)

    def register_buffer(self, name, tensor, persistent=True):
        return self._mixin_method('register_buffer', name, tensor, persistent)

    def register_parameter(self, name, param):
        return self._mixin_method('register_parameter', name, param)

    def extra_repr(self):
        return self._mixin_method('extra_repr')

    def children(self):
        return self._mixin_method('children')

    def modules(self):
        return self._mixin_method('modules')

    def _modules(self):
        return self._mixin_method('_modules')

    def _load_state_dict_post_hooks(self):
        hook_dict = self._mixin_method('_load_state_dict_post_hooks')
        if self.module_exists:

            for k, fn in hook_dict.items():
                def wrapper(module, *args, **kwargs):
                    return fn(module._module, *args, **kwargs)

                hook_dict[k] = wrapper

        return hook_dict

    def named_modules(self, *args, **kwargs):
        return self._mixin_method('named_modules', *args, **kwargs)

    def named_children(self) -> Iterator[Tuple[str, nn.Module]]:
        return self._mixin_method('named_children')

    def get_parameter(self, target: str) -> nn.Parameter:
        return self._mixin_method('get_parameter', target)

    def get_buffer(self, target: str) -> torch.Tensor:
        return self._mixin_method('get_buffer', target)

    def get_submodule(self, target: str) -> nn.Module:
        return self._mixin_method('get_submodule', target)

    def buffers(self, recurse: bool = True) -> Iterator[Tensor]:
        return self._mixin_method('buffers', recurse)

    def named_buffers(self, *arg, **kwargs) -> Iterator[Tuple[str, Tensor]]:
        return self._mixin_method('named_buffers', *arg, **kwargs)

    def register_module(self, *args, **kwargs) -> None:
        return self._mixin_method('register_module', *args, **kwargs)

    def to_empty(self, *args, **kwargs):
        return self._mixin_method('to_empty', *args, **kwargs)

    def add_module(self, name: str, module: Optional[nn.Module]) -> None:
        return self._mixin_method('add_module', name, module)

    def to(self, *args, **kwargs):
        if self.module_exists:
            self._module = self._module.to(*args, **kwargs)
        else:
            nn.Module.to(self, *args, **kwargs)
        return self

    def cuda(self, device: Optional[Union[int, device]] = None) -> nn.Module:
        if self.module_exists:
            self._module = self._module.cuda(device)
        else:
            nn.Module.cuda(self, device)
        return self

    def cpu(self) -> nn.Module:
        if self.module_exists:
            self._module = self._module.cpu()
        else:
            nn.Module.cpu(self)
        return self

    def type(self, dst_type: Optional[Union[str, device]] = None, *args, **kwargs) -> nn.Module:
        if self.module_exists:
            self._module = self._module.type(dst_type, *args, **kwargs)
        else:
            nn.Module.type(self, dst_type, *args, **kwargs)
        return self

    def float(self) -> nn.Module:
        if self.module_exists:
            self._module = self._module.float()
        else:
            nn.Module.float(self)
        return self

    def double(self) -> nn.Module:
        if self.module_exists:
            self._module = self._module.double()
        else:
            nn.Module.double(self)
        return self

    def half(self) -> nn.Module:
        if self.module_exists:
            self._module = self._module.half()
        else:
            nn.Module.half(self)
        return self

    def bfloat16(self) -> nn.Module:
        if self.module_exists:
            self._module = self._module.bfloat16()
        else:
            nn.Module.bfloat16(self)
        return self

    def __repr__(self):
        if self.module_exists:
            module_repr = repr(self._module)
            return f"BeamNN(wrapper)(\n{module_repr})"
        return f"BeamNN:{nn.Module.__repr__(self)}"

    @property
    def module_exists(self):
        if not hasattr(self, '_module') or self._module is None:
            return False
        return True

    def forward(self, *args, **kwargs):
        if self.module_exists:
            return self._module.forward(*args, **kwargs)
        raise NotImplementedError("Implement forward method in your BeamNN subclass")

    def __getattr__(self, item):
        if item != '_module' and self.module_exists:
            return getattr(self._module, item)
        return nn.Module.__getattr__(self, item)

    def __setattr__(self, key, value):
        if self.module_exists and not hasattr(self, key):
            return setattr(self._module, key, value)
        return nn.Module.__setattr__(self, key, value)

    @dynamo.disable
    def save_sample_input(self, *args, **kwargs):

        self._sample_input = {'args': recursive_clone(to_device(args, device='cpu')) if args else None,
                              'kwargs': recursive_clone(to_device(kwargs, device='cpu')) if kwargs else None,
                              'args_device': recursive_device(args) if args else None,
                              'kwargs_device': recursive_device(kwargs) if kwargs else None}

    @dynamo.disable
    def save_sample_output(self, output):
        self._sample_output = {'output': recursive_clone(to_device(output, device='cpu')),
                               'output_device': recursive_device(output)}

    def __call__(self, *args, **kwargs):

        if self._sample_input is None:
            self.save_sample_input(*args, **kwargs)

        if self.module_exists:
            output = self._module(*args, **kwargs)
        else:
            output = nn.Module.__call__(self, *args, **kwargs)

        if self._sample_output is None:
            self.save_sample_output(output)

        return output

    def optimize(self, method='compile', **kwargs):
        if method == 'compile':
            return BeamNN._compile(self, **kwargs)
        elif method == 'jit_trace':
            return BeamNN._jit_trace(self, **kwargs)
        elif method == 'jit_script':
            return BeamNN._jit_script(self)
        else:
            raise ValueError(f'Invalid optimization method: {method}, must be one of [compile|jit_trace|jit_script]')

    def export(self, path, method='onnx', **kwargs):
        if method == 'onnx':
            self._export_onnx(path, **kwargs)
        elif method == 'torchscript':
            self._export_jit_script(self)
        elif method == 'tensorrt':
            self._export_trt(path, **kwargs)
        else:
            raise ValueError(f'Invalid export method: {method}, must be one of [onnx]')

    def export_triton_server(self, path, method='onnx', model_name=None, model_version=None, **kwargs):

        from .triton import TritonConfig

        if model_version is None:
            model_version = '1'
        path = beam_path(path)
        if model_name is not None:
            path = path.joinpath(model_name)
        model_name = path.name
        path = path.joinpath(model_version)
        path.mkdir(parents=True, exist_ok=True)
        self.export(path.joinpath('model.onnx'), method=method, **kwargs)

        example_inputs, example_kwarg_inputs = self.example_input
        if example_inputs:
            max_batch_size = example_inputs[0].shape[0]
        elif example_kwarg_inputs:
            max_batch_size = next(example_kwarg_inputs.values()).shape[0]
        else:
            max_batch_size = 1

        config = TritonConfig(name=model_name, platform='pytorch_libtorch', max_batch_size=max_batch_size)
        config.save_to_file(path.joinpath('config.pbtxt'))

    @property
    def sample_input(self):
        return self._sample_input

    @property
    def sample_output(self):
        return self._sample_output

    @property
    def example_input(self):

        if self.sample_input['args'] is not None:
            example_inputs = recursive_clone(to_device(self.sample_input['args'],
                                                       device=self.sample_input['args_device']))
        else:
            example_inputs = None
        if self.sample_input['kwargs'] is not None:
            example_kwarg_inputs = recursive_clone(to_device(self.sample_input['kwargs'],
                                                         device=self.sample_input['kwargs_device']))
        else:
            example_kwarg_inputs = None
        return example_inputs, example_kwarg_inputs

    @classmethod
    def _jit_trace(cls, self, check_trace=None, check_inputs=None, check_tolerance=None, strict=None):

        self.assert_forward_was_executed('jit_trace')
        check_trace = check_trace or self.get_hparam('jit_check_trace', True)
        check_inputs = check_inputs or self.get_hparam('jit_check_inputs', None)
        check_tolerance = check_tolerance or self.get_hparam('jit_check_tolerance', 1e-5)
        strict = strict or self.get_hparam('jit_strict', True)

        module = self._module or self

        example_inputs, example_kwarg_inputs = self.example_input
        optimized = torch.jit.trace(module, example_inputs=example_inputs,
                               check_trace=check_trace, check_inputs=check_inputs, check_tolerance=check_tolerance,
                               strict=strict, example_kwarg_inputs=example_kwarg_inputs)

        return cls(_module=optimized, hparams=self.hparams, _model_type='torchscript')

    @classmethod
    def _jit_script(cls, self):

        self.assert_forward_was_executed('jit_script')
        example_inputs, example_kwarg_inputs = self.example_input
        if example_kwarg_inputs:
            raise NotImplementedError("JIT script does not support keyword arguments")

        module = self._module or self
        optimized = torch.jit.script(module, example_inputs=[example_inputs])

        return cls(_module=optimized, hparams=self.hparams, _model_type='torchscript')

    def assert_forward_was_executed(self, method_name):
        if self.sample_output is None or self.sample_input is None:
            raise ValueError(f"Please call the forward method at least once before calling {method_name}")

    @classmethod
    def _compile(cls, self, fullgraph=None, dynamic=None, backend=None,
                 mode=None, options=None, disable=False):

        fullgraph = fullgraph or self.get_hparam('compile_fullgraph', None)
        dynamic = dynamic or self.get_hparam('compile_dynamic', False)
        backend = backend or self.get_hparam('compile_backend', 'inductor')
        mode = mode or self.get_hparam('compile_mode', None)
        options = options or self.get_hparam('compile_options', None)

        module = self._module or self
        optimized = torch.compile(module, fullgraph=fullgraph, dynamic=dynamic, backend=backend,
                             mode=mode, options=options, disable=disable)

        return cls(_module=optimized, hparams=self.hparams, _model_type='compiled')

    def _export_jit_script(self, path):

        path = beam_path(path)
        with local_copy(path, as_beam_path=False) as tmp_path:
            if self._model_type == 'torchscript':
                self.module.save_torchscript(tmp_path)
            elif self._model_type == 'torch':
                model = self._jit_script(self).save(tmp_path)
                model.save_torchscript(tmp_path)
            else:
                raise NotImplementedError("Only TorchScript or native pytorch models can be exported with this method")

    def save_torchscript(self):
        if self.module_exists and self._model_type == 'torchscript':
            return self._module.save()
        raise NotImplementedError("Only TorchScript models can be saved with this method")

    def _export_trt(self, path,**kwargs):

        self._export_onnx(path, **kwargs)

        import tensorrt as trt
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        # Create a builder
        builder = trt.Builder(TRT_LOGGER)
        # Specify the builder configurations
        network = builder.create_network()
        parser = trt.OnnxParser(network, TRT_LOGGER)

        with local_copy(path, as_beam_path=True) as tmp_path:
            # Parse the model to create a network.
            parser.parse(tmp_path.read())
            # Build the engine
            engine = builder.build_cuda_engine(network)
            tmp_path.write_bin(engine.serialize())

    def _export_onnx(self, path, export_params=True, verbose=False, training='eval',
                     input_names=None, output_names=None, operator_export_type='ONNX', opset_version=None,
                     do_constant_folding=True, dynamic_axes=None, keep_initializers_as_inputs=None, custom_opsets=None,
                     export_modules_as_functions=False):

        self.assert_forward_was_executed('export(\'onnx\')')
        example_inputs, _ = self.example_input

        import torch.onnx

        if training == 'eval':
            training = torch.onnx.TrainingMode.EVAL
        elif training == 'train':
            training = torch.onnx.TrainingMode.TRAINING
        else:
            training = torch.onnx.TrainingMode.PRESERVE

        if operator_export_type == 'ONNX':
            operator_export_type = torch.onnx.OperatorExportTypes.ONNX
        elif operator_export_type == 'ONNX_ATEN':
            operator_export_type = torch.onnx.OperatorExportTypes.ONNX_ATEN
        elif operator_export_type == 'ONNX_FALLTHROUGH':
            operator_export_type = torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH
        else:
            raise ValueError(f'Invalid operator_export_type: {operator_export_type}, '
                             f'must be one of "ONNX", "ONNX_ATEN", or "ONNX_FALLTHROUGH"')

        path = beam_path(path)
        module = self._module or self
        with local_copy(path, as_beam_path=False) as tmp_path:
            torch.onnx.export(module, example_inputs, tmp_path, export_params=export_params,
                                     verbose=verbose, training=training, input_names=input_names,
                                     output_names=output_names, operator_export_type=operator_export_type,
                                     opset_version=opset_version, do_constant_folding=do_constant_folding,
                                     dynamic_axes=dynamic_axes, keep_initializers_as_inputs=keep_initializers_as_inputs,
                                     custom_opsets=custom_opsets,
                                     export_modules_as_functions=export_modules_as_functions)

    # add pruning and quantization methods
    # add methods for converting to other frameworks?


class BeamDDP(DDP):

    def __init__(self, module, *args, **kwargs):
        super().__init__(module, *args, **kwargs)
        self._init_is_done = True

    def __getattr__(self, item):
        if item.startswith('_') or item == '_init_is_done' or not hasattr(self, '_init_is_done'):
            return super().__getattr__(item)
        return getattr(self.module, item)

