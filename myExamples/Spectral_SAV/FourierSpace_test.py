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
        self.box = self.reset_box(box)
        self.N = N
        self.GD = len(N)

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
        if type(box) is list:
            newbox = np.array(box)
        elif (type(box) is np.ndarray) & (box.shape[0] > 1):
            newbox = box.flatten()
        else:
            newbox = box  # 即 box 为一维数组的情况
        return newbox

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

    def fourier_interpolation(self, data):
        """

        Parameters
        ----------
        data: data[0], data[1]
        """
        idx = data[0]
        idx[idx < 0] += self.N[0]
        F = self.function(dtype=data[1].dtype)

        F[tuple(idx.T)] = data[1]
        Err = self.fftn(F).real - np.fft.fftn(F).real
        e = np.max(np.abs(Err))
        return np.fft.fftn(F).real  # TODO: check here

    def fourier_interpolation1(self, data):
        """

        Parameters
        ----------
        data: data[0], data[1]
        """
        idx = data[0]
        idx[idx < 0] += self.N
        F = self.function(dtype=data[1].dtype)

        F[tuple(idx.T)] = data[1]
        Err = self.fftn(F).real - np.fft.fftn(F).real
        e = np.max(np.abs(Err))
        # return np.fft.fftn(F).real
        return self.fftn(F).real

    def function_norm(self, u):
        val = np.sqrt(np.sum(self.ifftn(u)) ** 2).real
        return val

    def interpolation(self, u):
        p = self.interpolation_points()
        return u(p)

    def reciprocal_lattice(self, project_matrix=None, sparse=True,
                           return_square=False):
        """
        倒易空间的网格
        """
        N = self.N
        GD = self.GD
        box = self.box

        f = self.fftfreq(N) * N
        f = np.meshgrid(*(GD * (f,)), sparse=sparse)
        rBox = 2 * np.pi * inv(box).T
        n = GD
        if project_matrix is not None:
            n = project_matrix.shape[0]
            rBox = project_matrix @ rBox
        k = []
        for i in range(n):
            k.append(sum(map(lambda x: x[0] * x[1], zip(rBox[i], f))))

        if return_square:
            return k, sum(map(lambda x: x ** 2, k))
        else:
            return k

    def linear_equation_fft_solver(self, f,
                                   cfun=lambda x: 1 + sum(map(lambda y: y ** 2, x))):
        N = self.N
        GD = self.GD
        xi = self.reciprocal_lattice()
        F = self.interpolation(f)
        F = self.ifftn(F)
        U = F / cfun(xi)
        U = self.fftn(U).real
        return U

    def function(self, dim=None, dtype=None):
        dtype = self.ftype if dtype is None else dtype
        N = self.N
        GD = self.GD
        box = self.box
        shape = GD * (N,)
        if dim is not None:
            shape = (dim,) + shape
        f = np.zeros(shape, dtype=dtype)
        return f

    def error(self, u, U):
        N = self.N
        GD = self.GD
        U0 = self.interpolation(u)
        error = np.sqrt(np.sum((U0 - U) ** 2) / N ** GD)
        return error
