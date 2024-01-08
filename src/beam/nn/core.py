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
    def from_module(cls, module):

        """
        Class method to create a BeamNN object from an existing nn.Module.

        Args:
            module (nn.Module): The PyTorch module to be wrapped.

        Returns:
            BeamNN: A new BeamNN instance wrapping the provided module.
        """

        if isinstance(module, cls):
            beam_module = module
        else:
            beam_module = cls(_module=module)
        return beam_module

    def parameters(self, *args, **kwargs):
        if self._module is not None:
            return self._module.parameters(*args, **kwargs)
        else:
            return nn.Module.parameters(self, *args, **kwargs)

    def named_parameters(self, *args, **kwargs):
        if self._module is not None:
            return self._module.named_parameters(*args, **kwargs)
        else:
            return nn.Module.named_parameters(self, *args, **kwargs)

    def state_dict(self, *args, **kwargs):
        if self._module is not None:
            return self._module.state_dict(*args, **kwargs)
        else:
            return nn.Module.state_dict(self, *args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        if self._module is not None:
            self._module.load_state_dict(state_dict, *args, **kwargs)
        else:
            nn.Module.load_state_dict(self, state_dict, *args, **kwargs)

    def apply(self, fn):
        if self._module is not None:
            self._module.apply(fn)
        else:
            nn.Module.apply(self, fn)

    def register_buffer(self, name, tensor, persistent=True):
        if self._module is not None:
            self._module.register_buffer(name, tensor, persistent)
        else:
            nn.Module.register_buffer(self, name, tensor, persistent)

    def register_parameter(self, name, param):
        if self._module is not None:
            self._module.register_parameter(name, param)
        else:
            nn.Module.register_parameter(self, name, param)

    def extra_repr(self):
        if self._module is not None:
            return self._module.extra_repr()
        else:
            return nn.Module.extra_repr(self)

    def children(self):
        if self._module is not None:
            return self._module.children()
        else:
            return nn.Module.children(self)

    def modules(self):
        if self._module is not None:
            for module in self._module.modules():
                yield module
        else:
            for module in nn.Module.modules(self):
                yield module

    def named_modules(self, *args, **kwargs):
        if self._module is not None:
            for name, module in self._module.named_modules(*args, **kwargs):
                yield name, module
        else:
            for name, module in nn.Module.named_modules(self, *args, **kwargs):
                yield name, module

    def named_children(self) -> Iterator[Tuple[str, nn.Module]]:
        if self._module is not None:
            for name, module in self._module.named_children():
                yield name, module
        else:
            for name, module in nn.Module.named_children(self):
                yield name, module

    def get_parameter(self, target: str) -> nn.Parameter:
        if self._module is not None:
            return self._module.get_parameter(target)
        else:
            return nn.Module.get_parameter(self, target)

    def get_buffer(self, target: str) -> torch.Tensor:
        if self._module is not None:
            return self._module.get_buffer(target)
        else:
            return nn.Module.get_buffer(self, target)

    def get_submodule(self, target: str) -> nn.Module:
        if self._module is not None:
            return self._module.get_submodule(target)
        else:
            return nn.Module.get_submodule(self, target)

    def buffers(self, recurse: bool = True) -> Iterator[Tensor]:
        if self._module is not None:
            return self._module.buffers(recurse)
        else:
            return nn.Module.buffers(self, recurse)

    def named_buffers(self, *arg, **kwargs) -> Iterator[Tuple[str, Tensor]]:
        if self._module is not None:
            return self._module.named_buffers(*arg, **kwargs)
        else:
            return nn.Module.named_buffers(self, *arg, **kwargs)

    def register_module(self, *args, **kwargs) -> None:
        if self._module is not None:
            self._module.register_module(*args, **kwargs)
        else:
            nn.Module.register_module(self, *args, **kwargs)

    def get_extra_state(self, *args, **kwargs):
        if self._module is not None:
            return self._module.get_extra_state(*args, **kwargs)
        else:
            return nn.Module.get_extra_state(self, *args, **kwargs)

    def set_extra_state(self, *args, **kwargs):
        if self._module is not None:
            return self._module.set_extra_state(*args, **kwargs)
        else:
            return nn.Module.set_extra_state(self, *args, **kwargs)

    def to_empty(self, *args, **kwargs):
        if self._module is not None:
            return self._module.to_empty(*args, **kwargs)
        else:
            return nn.Module.to_empty(self, *args, **kwargs)

    def add_module(self, name: str, module: Optional['Module']) -> None:
        if self._module is not None:
            self._module.add_module(name, module)
        else:
            nn.Module.add_module(self, name, module)

    def __repr__(self):
        module_repr = repr(self._module) if self._module is not None else nn.Module.__repr__(self)
        return f"BeamNN({module_repr})"

    def forward(self, *args, **kwargs):
        if self._module is not None:
            return self._module.forward(*args, **kwargs)
        else:
            raise NotImplementedError("Implement forward method in your BeamNN subclass")

    def __getattr__(self, item):
        if self._module is None:
            return nn.Module.__getattr__(self, item)
        return getattr(self._module, item)

    def __setattr__(self, key, value):
        if self._module is None:
            return nn.Module.__setattr__(self, key, value)
        return setattr(self._module, key, value)

    def __call__(self, *args, **kwargs):

        if self._sample_input is None:
            self._sample_input = {'args': recursive_clone(to_device(args, device='cpu')),
                                  'kwargs': recursive_clone(to_device(kwargs, device='cpu'))}

        if self._module is None:
            return nn.Module.__call__(self, *args, **kwargs)
        return self._module(*args, **kwargs)

    def optimize(self, method='compile', **kwargs):
        if method == 'compile':
            return self._compile(**kwargs)
        elif method == 'jit_trace':
            return self._jit_trace(**kwargs)
        elif method == 'jit_script':
            return self._jit_script(**kwargs)
        elif method == 'onnx':
            return self._onnx(**kwargs)
        else:
            raise ValueError(f'Invalid optimization method: {method}, must be one of "compile", "jit_trace", '
                             f'"jit_script", or "onnx"')

    @property
    def sample_input(self):
        return self._sample_input

    def _jit_trace(self, optimize=None, check_trace=True, check_inputs=None, check_tolerance=1e-5, strict=True):
        return torch.jit.trace(self, example_inputs=self.sample_input['args'], optimize=optimize,
                               check_trace=check_trace, check_inputs=check_inputs, check_tolerance=check_tolerance,
                               strict=strict, example_kwarg_inputs=self.sample_input['kwargs'])

    def _jit_script(self, optimize=None):
        return torch.jit.script(self, optimize=optimize,
                                example_inputs=(self.sample_input['args'], self.sample_input['kwargs']))

    def _compile(self, fullgraph=False, dynamic=False, backend="inductor",
                 mode=None, options=None, disable=False):
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
        disable = path.scheme == 'file'

        with local_copy(path, disable=disable) as tmp_path:
            torch.onnx.export(self, self.sample_input['args'], tmp_path, export_params=export_params,
                                     verbose=verbose, training=training, input_names=input_names,
                                     output_names=output_names, operator_export_type=operator_export_type,
                                     opset_version=opset_version, do_constant_folding=do_constant_folding,
                                     dynamic_axes=dynamic_axes, keep_initializers_as_inputs=keep_initializers_as_inputs,
                                     custom_opsets=custom_opsets,
                                     export_modules_as_functions=export_modules_as_functions)

    # add pruning and quantization methods
    # add methods for converting to other frameworks?
