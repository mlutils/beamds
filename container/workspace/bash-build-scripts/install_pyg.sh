
# RUN pip install -U -r /workspace/pip/requirements_torch_geometric.txt
# RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# Automatically detect CUDA architecture
#CUDA_ARCH=$(python -c "import torch; cap=torch.cuda.get_device_capability(); print(f'{cap[0]}.{cap[1]}') if torch.cuda.is_available() else exit(1)")
export TORCH_CUDA_ARCH_LIST=$CUDA_CAPABILITIES

echo "Detected CUDA Arch: $TORCH_CUDA_ARCH_LIST"

pip install --verbose git+https://github.com/pyg-team/pyg-lib.git
pip install --verbose torch_scatter
pip install --verbose torch_sparse
pip install --verbose torch_cluster
pip install --verbose torch_spline_conv

pip install torch_geometric
pip install torchpq