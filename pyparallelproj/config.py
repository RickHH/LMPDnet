from pathlib import Path

import cupy as cp

with open(Path(__file__).parents[1] / 'cuda' / 'lcx_projector_kernels.cu', 'r') as f:
    lines = f.read()
    
    compute_systemG_kernel = cp.RawKernel(lines, 'compute_systemG_kernel')


