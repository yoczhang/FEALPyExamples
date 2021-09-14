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
from fealpy.functionspace.FourierSpace import FourierSpace
import pyfftw as pw


class SAVCHModel:
    def __init__(self, pde, box, dt):
        self.space = FourierSpace(box, pde.N)
        self.GD = self.space.GD
        self.pde = pde
        self.N = pde.N
        self.T = pde.T
        self.timemesh, self.dt = self.pde.time_mesh(dt)

    def SAVSolver(self):
        pde = self.pde
        space = self.space

        N = pde.N
        h = space.h
        T = pde.T
        dt = self.dt
        timemesh = self.timemesh

        epsilon = pde.epsilon
        gamma = pde.gamma
        beta = pde.beta
        alpha = pde.alpha

        # # Initial value u0 (t = 0)
        uin = 0.05 * (2 * np.random.rand(*N) - 1)
        uaver = np.sum(uin) / np.prod(N)
        u0 = uin - uaver

        # # setting the initial something
        NT = len(timemesh)
        U = 1./epsilon**2 * u0 * (u0**2 - 1)
        E1 = 1. / (4 * epsilon ** 2) * space.DFIntegral(u0**2 - 1, u0**2 - 1)

        # # the temporary variable
        rn = np.sqrt(E1)
        bn = U/rn
        Gbn = space.DFTLaplace(bn)
        cn = u0 + dt*rn*Gbn - dt/2*space.DFIntegral(bn, u0)*Gbn
        Dlaplace = np.sum(space.FourierDiffCoeff(2), axis=0)  # (Nx,Ny)
        DFT_I = np.ones(*N)
        DFT_G = Dlaplace
        DFT_L = - Dlaplace

        DFT_M = DFT_I - dt*DFT_G*DFT_L
        Gbx = space.ifftn(space.fftn(Gbn) / DFT_M)  # (Nx,Ny)
        gamma = - space.DFIntegral(bn, Gbx)

        cx = space.ifftn(space.fftn(cn) / DFT_M)

        timeCount = np.zeros((NT,))
        storeEnergy = np.zeros((NT,))











