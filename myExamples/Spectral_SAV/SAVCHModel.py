#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: SAVCHModel.py
# @Author: Yongchao Zhang, Northwest University, Xi'an
# @E-mail: yoczhang@nwu.edu.cn
# @Site:
# @Time: Jun 18, 2021
# ---


import numpy as np
from FourierSpace_test import FourierSpace
# from fealpy.functionspace.FourierSpace import FourierSpace
import pyfftw as pw


class SAVCHModel:
    def __init__(self, PDE, box):
        self.space = FourierSpace(box, PDE.N)
        self.GD = self.space.GD
        self.PDE = PDE
        self.N = PDE.N
        self.T = PDE.T
        self.dt = PDE.dt

    def solve(self):
        PDE = self.PDE

        N = PDE.N
        h = PDE.h
        T = PDE.T
        dt = PDE.dt

        epsilon = PDE.epsilon
        gamma = PDE.gamma
        beta = PDE.beta
        alpha = PDE.alpha

        # # Initial value u0 (t = 0)
        uin = 0.05 * (2 * np.random.rand(N, N) - 1)
        uaver = np.sum(uin) / N ** 2
        u0 = uin - uaver

        # # setting the initial something
        NT = int(T/dt)
        U = 1. / epsilon ** 2 * u0 * (u0 ** 2 - 1 - beta)
        E = 1. / (4 * epsilon ** 2) * h.prod() * np.sum((u0 ** 2 - 1 - beta) ** 2)

        # # the temporary variable
        rn = np.sqrt(E)
        bn = U/rn

        timeCount = np.zeros((NT,))
        storeEnergy = np.zeros((NT,))

    def fourierDiffCoeff(self, m):
        """
        returns the m-th derivative of function
        :param m:
        :return:
        """

        N = self.N  # TODO: 这里规定 N 为一维数为 GD 的数组, 在每个方向为 N
        space = self.space
        box = space.box  # box.shape: (GD, 2),
        GD = space.GD
        multipleN = np.ones((GD,)) * N  # 分别在 x, y, z 方向上给出采样点个数 Nx, Ny, Nz
        L = np.abs(box[:, 1] - box[:, 0])  # L.shape: (GD,), 用来存储 x, y, z 方向上的区间长度
        h = L / multipleN  # 用来存储 x, y, z 方向上的网格尺寸
        normalization = 2*np.pi / L  # normalization.shape: (GD,), 在 x, y, z 方向上, 将 [a, b] 映射到 [0, 2*pi]

        normalK = []
        for i in range(GD):
            basicK = 1j*np.concatenate([np.arange(0, multipleN[i]/2+1), np.arange(-multipleN[i]/2+1, 0)])
            normalK.append(basicK * normalization[i])

        # if GD == 2:
        #     Kx, Ky = np.meshgrid(normalK[0]**m, normalK[1]**m)
        #     print(np.allclose(rr, [Kx, Ky]))
        #     return Kx, Ky
        # elif GD == 3:
        #     Kx, Ky, Kz = np.meshgrid(normalK[0]**m, normalK[1]**m, normalK[2]**m)
        #     return Kx, Ky, Kz
        # else:
        #     raise ValueError("The dimension of space is false")

        meshgridK = np.meshgrid(*normalK)
        return list(map(lambda x: x**m, meshgridK))

    def DFTLaplace(self, phi):
        pass








