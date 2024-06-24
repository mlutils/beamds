from beam.nn import LinearNet
from beam.nn.core import BeamNN
from beam import beam_path
import torch


f = LinearNet(100)
g = BeamNN.from_module(f)
path = beam_path('/tmp/net')
y = g(torch.randn(16, 100))

g.optimize('onnx', path)
