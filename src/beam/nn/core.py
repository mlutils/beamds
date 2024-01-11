from typing import Iterator, Tuple, Optional, Union

import torch
from torch import nn, Tensor, device
from torch.nn.modules.module import T

from ..core import Processor
from ..utils import recursive_clone, to_device
from ..path import beam_path, local_copy


class BeamNN(nn.Module, Processor):
    """
    BeamNN is a wrapper class around PyTorch's nn.Module, with added functionalities
    from the Processor class. It allows dynamic integration of an existing nn.Module
    instance with additional processing capabilities.

    Attributes:
        _sample_input: A sample input for JIT tracing or other optimization methods.
        _module: The wrapped nn.Module instance.
    """

    def __init__(self, *args, _module=None, **kwargs):

        """
        Initialize the BeamNN wrapper.

        Args:
            _module (nn.Module): The PyTorch module to be wrapped.
            *args, **kwargs: Additional arguments for the Processor initialization.
        """

        nn.Module.__init__(self)
        Processor.__init__(self, *args, **kwargs)
        self._sample_input = None
        self._module = _module

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
        return self._mixin_method('to_empty')(self, *args, **kwargs)

    def add_module(self, name: str, module: Optional[nn.Module]) -> None:
        return self._mixin_method('add_module', name, module)

    def __repr__(self):
        if self.module_exists:
            module_repr = repr(self._module)
            return f"BeamNN(wrapper)(\n{module_repr})"
        return nn.Module.__repr__(self)

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

    def __call__(self, *args, **kwargs):

        if self._sample_input is None:
            self._sample_input = {'args': recursive_clone(to_device(args, device='cpu')),
                                  'kwargs': recursive_clone(to_device(kwargs, device='cpu'))}

        if self.module_exists:
            return self._module(*args, **kwargs)
        return nn.Module.__call__(self, *args, **kwargs)

    def optimize(self, method='compile', *args, **kwargs):
        if method == 'compile':
            return self._compile(*args, **kwargs)
        elif method == 'jit_trace':
            return self._jit_trace(*args, **kwargs)
        elif method == 'jit_script':
            return self._jit_script(*args, **kwargs)
        elif method == 'onnx':
            return self._onnx(*args, **kwargs)
        else:
            raise ValueError(f'Invalid optimization method: {method}, must be one of "compile", "jit_trace", '
                             f'"jit_script", or "onnx"')

    @property
    def sample_input(self):
        return self._sample_input

    def _jit_trace(self, check_trace=None, check_inputs=None, check_tolerance=None, strict=None):

        check_trace = check_trace or self.get_hparam('jit_check_trace', True)
        check_inputs = check_inputs or self.get_hparam('jit_check_inputs', None)
        check_tolerance = check_tolerance or self.get_hparam('jit_check_tolerance', 1e-5)
        strict = strict or self.get_hparam('jit_strict', True)

        return torch.jit.trace(self, example_inputs=self.sample_input['args'],
                               check_trace=check_trace, check_inputs=check_inputs, check_tolerance=check_tolerance,
                               strict=strict, example_kwarg_inputs=self.sample_input['kwargs'])

    def _jit_script(self):
        if self.sample_input['kwargs']:
            raise NotImplementedError("JIT script does not support keyword arguments")
        return torch.jit.script(self, example_inputs=[self.sample_input['args']])

    def _compile(self, fullgraph=None, dynamic=None, backend=None,
                 mode=None, options=None, disable=False):

        fullgraph = fullgraph or self.get_hparam('compile_fullgraph', None)
        dynamic = dynamic or self.get_hparam('compile_dynamic', False)
        backend = backend or self.get_hparam('compile_backend', 'inductor')
        mode = mode or self.get_hparam('compile_mode', None)
        options = options or self.get_hparam('compile_options', None)

        return torch.compile(self, fullgraph=fullgraph, dynamic=dynamic, backend=backend,
                             mode=mode, options=options, disable=disable)

    def _onnx(self, path, export_params=True, verbose=False, training='eval',
              input_names=None, output_names=None, operator_export_type='ONNX', opset_version=None,
              do_constant_folding=True, dynamic_axes=None, keep_initializers_as_inputs=None, custom_opsets=None,
              export_modules_as_functions=False):




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
        with local_copy(path, as_beam_path=False) as tmp_path:
            torch.onnx.export(self, self.sample_input['args'], tmp_path, export_params=export_params,
                                     verbose=verbose, training=training, input_names=input_names,
                                     output_names=output_names, operator_export_type=operator_export_type,
                                     opset_version=opset_version, do_constant_folding=do_constant_folding,
                                     dynamic_axes=dynamic_axes, keep_initializers_as_inputs=keep_initializers_as_inputs,
                                     custom_opsets=custom_opsets,
                                     export_modules_as_functions=export_modules_as_functions)

    # add pruning and quantization methods
    # add methods for converting to other frameworks?
