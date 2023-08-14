import math
from pathlib import Path

import cupy as cp
import numpy as np

with open(Path(__file__).parents[1] / 'cuda' / 'lcx_projector_kernels.cu', 'r', encoding='utf-8') as f:
    lines = f.read()
    lcx_fwd_tof_lm_cuda_kernel = cp.RawKernel(lines, 'lcx_fwd_tof_lm_cuda_kernel')
    lcx_back_tof_lm_cuda_kernel = cp.RawKernel(lines, 'lcx_back_tof_lm_cuda_kernel')
    compute_systemG_kernel = cp.RawKernel(lines, 'compute_systemG_kernel')
    compute_systemG_nontof_kernel = cp.RawKernel(lines, 'compute_systemG_nontof_kernel')


def compute_systemG_lm(xstart,
                   xend,
                   img_origin,
                   voxsize,
                   sysG,
                   nLORs,
                   img_dim,
                   tofbin_width,
                   sigma_tof,
                   tofcenter_offset,
                   nsigmas,
                   tofbin,
                   nevents,
                   threadsperblock):

    ok = compute_systemG_kernel(
            (math.ceil(nLORs / threadsperblock), ), (threadsperblock, ),
            (xstart.ravel(), xend.ravel(),
            cp.asarray(img_origin), cp.asarray(voxsize), sysG,
            np.int64(nLORs), cp.asarray(img_dim),
            np.float32(tofbin_width), cp.asarray(sigma_tof).ravel(),
            cp.asarray(tofcenter_offset).ravel(), np.float32(nsigmas),
            tofbin, nevents)
    )

    return ok

def compute_systemG_nontof_lm(xstart,
                              xend,
                              img_origin,
                              voxsize,
                              sysG,
                              nLORs,
                              img_dim,
                              threadsperblock=64):
        ok = compute_systemG_nontof_kernel(
                (math.ceil(nLORs / threadsperblock), ), (threadsperblock, ),
                (xstart.ravel(), xend.ravel(), 
                cp.asarray(img_origin), cp.asarray(voxsize), sysG,
                np.int64(nLORs), cp.asarray(img_dim))
        )

        return ok

