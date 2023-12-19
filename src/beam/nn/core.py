import torch
from torch import nn

from ..core import Processor
from ..utils import recursive_clone, to_device
from ..path import beam_path, local_copy


class BeamNN(nn.Module, Processor):

    def __init__(self, *args, **kwargs):
        nn.Module.__init__(self)
        Processor.__init__(self, *args, **kwargs)
        self._sample_input = None

    @classmethod
    def from_module(cls, module, *args, **kwargs):
        beam_module = cls(*args, **kwargs)

        for k, v in beam_module.__dict__.items():
            if k not in module.__dict__:
                module.__dict__[k] = v

        module.__call__ = beam_module.__call__.__get__(module, cls)
        module.optimize = beam_module.optimize.__get__(module, cls)
        module._sample_input = beam_module._sample_input
        module._jit_trace = beam_module._jit_trace.__get__(module, cls)
        module._jit_script = beam_module._jit_script.__get__(module, cls)
        module._compile = beam_module._compile.__get__(module, cls)
        module._onnx = beam_module._onnx.__get__(module, cls)
        module.sample_input = beam_module.sample_input

        return module

    def __call__(self, *args, **kwargs):

        if self._sample_input is None:
            self._sample_input = {'args': recursive_clone(to_device(args, device='cpu')),
                                  'kwargs': recursive_clone(to_device(kwargs, device='cpu'))}

        return super().__call__(*args, **kwargs)

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
