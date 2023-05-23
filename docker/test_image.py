
# Test if the docker image is working properly
import beam
print("Beam version: ", beam.__version__)

import torch
print("Pytorch version: ", torch.__version__)
print(torch.randn(100).to(0))

# test ray
from beam.utils import find_port
import ray
runtime_env = {"working_dir": ".." }
ray.init(runtime_env=runtime_env, dashboard_port=int(find_port(application='ray')), include_dashboard=True)

# test torch_geometric
import torch_geometric as tg
f = tg.nn.GATConv(10, 10)
f.to(0)

