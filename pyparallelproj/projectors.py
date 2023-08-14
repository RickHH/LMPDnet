import ctypes
import math
import os

import cupy as cp
import numpy as np
import numpy.ctypeslib as npct
import scipy.io as sio

from .wrapper import compute_systemG_lm, compute_systemG_nontof_lm


class Projector:
    def __init__(self, scanner, sino_params, img_dim, img_origin=None, voxsize=np.ones(3), 
                 sigma_tof=60./2.35, n_sigmas=3, threadsperblock=64, devicenum=0):
        self.scanner = scanner
        self.sino_params = sino_params
        self.img_dim = img_dim                                 
        if not isinstance(self.img_dim, np.ndarray):
            self.img_dim = np.array(img_dim)
        self.nvox = np.prod(self.img_dim)
        self.img_dim = self.img_dim.astype(ctypes.c_int)

        self.voxsize = voxsize.astype(np.float32)

        if img_origin is None:
            self.img_origin = (-(self.img_dim / 2) + 0.5) * self.voxsize
        else:
            self.img_origin = img_origin
        self.img_origin = self.img_origin.astype(np.float32)

        # tof parameters
        self.sigma_tof = sigma_tof
        self.nsigmas = float(n_sigmas)
        self.tofbin_width = self.sino_params.tofbin_width
        self.ntofbins = self.sino_params.ntofbins

        # gpu parameters (not relevant when not run on gpu)
        self.threadsperblock = threadsperblock
        self.devicenum = devicenum
    
    def computesysG(self, events):
        with cp.cuda.Device(self.devicenum):            
            nlors = events.shape[0]
            N = int(nlors * self.nvox)
            sysG = cp.zeros(N, dtype=cp.float32)

            xstart = self.scanner.get_crystal_coordinates(events[:, 0:2])
            xend = self.scanner.get_crystal_coordinates(events[:, 2:4])

            sigma_tof = cp.array([self.sigma_tof], dtype=cp.float32)
            tofcenter_offset = cp.zeros(1, dtype=cp.float32)

            tofbin = cp.asarray(events[:, 4].astype(cp.int16))
            nevents = cp.asarray(events[:, 5].astype(cp.int16))

            compute_systemG_lm(xstart.ravel(),
                        xend.ravel(),
                        self.img_origin,
                        self.voxsize,
                        sysG,
                        nlors,
                        self.img_dim,
                        self.tofbin_width,
                        sigma_tof.ravel(),
                        tofcenter_offset.ravel(),
                        self.nsigmas,
                        tofbin,
                        nevents,
                        threadsperblock = self.threadsperblock)

            sysG = cp.reshape(sysG, (nlors, self.nvox))

        return sysG


    def computesysGnontof(self, events):
        with cp.cuda.Device(self.devicenum):

            nlors = events.shape[0]

            sysG = cp.zeros(nlors * self.nvox, dtype=cp.float32)

            xstart = self.scanner.get_crystal_coordinates(events[:, 0:2])
            xend = self.scanner.get_crystal_coordinates(events[:, 2:4])

            nevents = cp.asarray(events[:, 5].astype(cp.int16))

            compute_systemG_nontof_lm(xstart.ravel(),
                                        xend.ravel(),
                                        self.img_origin,
                                        self.voxsize,
                                        sysG,
                                        nlors,
                                        self.img_dim,
                                        threadsperblock = self.threadsperblock)

            sysG = cp.reshape(sysG, (nlors, self.nvox))

        return sysG        


    def fwd_project_lm(self,
                       img,
                       events,
                       tofcenter_offset=None,
                       sigma_tof_per_lor=None):

        with cp.cuda.Device(self.devicenum):
            if not img.dtype is cp.dtype('float32'):
                img = img.astype(cp.float32)

            nevents = events.shape[0]

            img_fwd = cp.zeros(nevents, dtype=cp.float32)

            xstart = self.scanner.get_crystal_coordinates(events[:, 0:2])
            xend = self.scanner.get_crystal_coordinates(events[:, 2:4])

            sigma_tof = cp.array([self.sigma_tof], dtype=cp.float32)


            tofcenter_offset = cp.zeros(1, dtype=cp.float32)

            tofbin = cp.asarray(events[:, 4].astype(cp.int16))                   

            ok = lcx_fwd_tof_lm(xstart.ravel(),
                                xend.ravel(),
                                img.ravel(),
                                self.img_origin,
                                self.voxsize,
                                img_fwd,
                                nevents,
                                self.img_dim,
                                self.tofbin_width,
                                sigma_tof.ravel(),
                                tofcenter_offset,
                                self.nsigmas,
                                tofbin,
                                threadsperblock=self.threadsperblock)

        return img_fwd

    def back_project_lm(self,
                        values,
                        events,
                        tofcenter_offset=None,
                        sigma_tof_per_lor=None):

        with cp.cuda.Device(self.devicenum):
            values = values.astype(cp.float32)

            nevents = events.shape[0]

            back_img = cp.zeros(int(self.nvox), dtype=cp.float32)

            xstart = self.scanner.get_crystal_coordinates(events[:, 0:2])
            xend = self.scanner.get_crystal_coordinates(events[:, 2:4])

            # TOF back projection

            sigma_tof = cp.array([self.sigma_tof],dtype=cp.float32)


            tofcenter_offset = cp.zeros(1, dtype=cp.float32)

            tofbin = cp.asarray(events[:, 4].astype(np.int16))


            ok = lcx_back_tof_lm(xstart.ravel(),
                                        xend.ravel(),
                                        back_img,
                                        self.img_origin,
                                        self.voxsize,
                                        values.ravel(),
                                        nevents,
                                        self.img_dim,
                                        self.tofbin_width,
                                        sigma_tof.ravel(),
                                        tofcenter_offset,
                                        self.nsigmas,
                                        tofbin,
                                        threadsperblock=self.threadsperblock)


        return back_img.reshape(self.img_dim)