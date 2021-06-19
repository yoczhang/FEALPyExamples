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


class SAVCHModel:
    def __init__(self, PDE, box):
        self.FSpace = FourierSpace(box, PDE.N)
        self.PDE = PDE
        self.N = PDE.N
        self.h = PDE.h
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
        TN = int(T/dt)
        U = 1. / epsilon ** 2 * u0 * (u0 ** 2 - 1 - beta)
        E = 1. / (4 * epsilon ** 2) * h ** 2 * np.sum((u0 ** 2 - 1 - beta) ** 2)

        # # the temporary variable
        rn = np.sqrt(E)
        bn = U/rn



        timeCount = np.zeros((TN,))
        storeEnergy = np.zeros((TN,))


    def dftLaplace(self):
        N = self.N


