#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: FourierSpace_test.py
# @Author: Yongchao Zhang, Northwest University, Xi'an
# @E-mail: yoczhang@nwu.edu.cn
# @Site:
# @Time: Jun 17, 2021
# ---

__doc__ = """
This file is the test-version from fealpy.functionspace.FourierSpace
"""

import numpy as np
from numpy.linalg import inv

import scipy.fftpack as spfft
import pyfftw


class FourierSpace:
    def __init__(self, box, N, dft=None):
        self.box, self.L = self.reset_box(box)
        self.N = N
        self.GD = len(N)
        self.h = self.L / np.array(N)

        self.ftype = np.float
        self.itype = np.int32

        if dft is None:
            ncpt = np.array(N)
            a = pyfftw.empty_aligned(ncpt, dtype=np.complex128)
            self.fftn = pyfftw.builders.fftn(a)
            b = pyfftw.empty_aligned(ncpt, dtype=np.complex128)
            self.ifftn = pyfftw.builders.ifftn(b)
            self.fftfreq = spfft.fftfreq  # TODO:change to pyfftw
            self.fftfreq = pyfftw.interfaces.scipy_fftpack.fftfreq
        elif dft == "scipy":
            self.fftn = spfft.fftn
            self.ifftn = spfft.ifftn
            self.fftfreq = spfft.fftfreq
        else:
            self.fftn = dft.fftn
            self.ifftn = dft.ifftn
            self.fftfreq = dft.fftfreq

    def reset_box(self, box):
        """
        In this function, we translate 'box' into one-dimension numpy's array
        :param box:
        :return: newbox
        """
        if type(box) is list:
            newbox = np.array(box)
        elif (type(box) is np.ndarray) & (box.shape[0] > 1):
            newbox = box.flatten()
        else:
            newbox = box  # 即 box 为一维数组的情况
        L = newbox[1::self.GD] - newbox[::self.GD]
        return newbox, L

    def number_of_dofs(self):
        return np.prod(np.array(self.N))

    def interpolation_points(self):
        N = self.N
        GD = self.GD
        box = self.box

        coord_1d = []
        for i in range(GD):
            coord_1d.append(np.arange(box[i*GD], box[i*GD+1], abs(box[i*GD]-box[i*GD+1])/N[i]))
        coord_list = np.meshgrid(*coord_1d)  # list: (GD,)

        switch_idx = [i for i in range(1, GD+1)]
        switch_idx.append(0)
        coord_array = np.array(coord_list).transpose(switch_idx) # shape: (...,GD). Such as: GD=2, [...,0] is the x-coord, [...,1] is the y-coord
        return coord_array

    def function_norm(self, u):
        val = np.sqrt(np.sum(self.ifftn(u)) ** 2).real
        return val

    def interpolation(self, u):
        p = self.interpolation_points()
        return u(p)

    def function(self, dim=None, dtype=None):
        dtype = self.ftype if dtype is None else dtype
        shape = self.N
        if dim is not None:
            shape = (dim,) + shape  # Here, dim denotes the dim-vector functions
        f = np.zeros(shape, dtype=dtype)
        return f

    def error(self, u, U):
        N = self.N
        GD = self.GD
        U0 = self.interpolation(u)
        error = np.sqrt(np.sum((U0 - U) ** 2) / N ** GD)
        return error

    def FourierDiffCoeff(self, m):
        """
        :param m: the m-th order differential
        :return: the Fourier-coefficients of m-th differential
        """

        multipleN = np.array(self.N)  # 分别在 x, y, z 方向上给出采样点个数 Nx, Ny, Nz
        box = self.box  # box.shape: (GD, 2),
        GD = self.GD
        # if (type(box) is np.ndarray) & (len(box) == GD):
        #     L = np.abs(box[:, 1] - box[:, 0])  # L.shape: (GD,), 用来存储 x, y, z 方向上的区间长度
        # elif type(box) is list:
        #     L = np.array(box[1::GD])-np.array(box[::GD])
        # else:
        #     L = multipleN
        # L = np.array(box[1::GD]) - np.array(box[::GD])
        # h = L / multipleN  # 用来存储 x, y, z 方向上的网格尺寸
        L = self.L
        normalization = 2*np.pi / L  # normalization.shape: (GD,), 在 x, y, z 方向上, 将 [a, b] 映射到 [0, 2*pi]

        normalK = []
        for i in range(GD):
            # # Wavenumbers (with ordering suitable for Numpy's FFT)
            basicK = 1j*np.concatenate([np.arange(0, multipleN[i]/2+1), np.arange(-multipleN[i]/2+1, 0)])
            normalK.append(basicK * normalization[i])

        meshgridK = np.meshgrid(*normalK)
        return list(map(lambda x: x**m, meshgridK))
