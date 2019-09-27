#!/anaconda3/envs/FEALPy/bin python3.7
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: Vcycle.py
# @Author: Yongchao Zhang
# @E-mail: yoczhang@126.com
# @Time: Sep 22, 2019
# ---

import numpy as np
from timeit import default_timer as timer
from Smoothers_2D import DGS_smoother
from getPDEdata import getPDEBasicData


class VCYCLE:
    def __init__(self, pde, ncycle):
        self.pde = pde
        self.h = pde.h
        self.ncycle = ncycle  # this gives the number of V-cycle
        self.DGS_obj = DGS_smoother(pde)

    def Vcycle(self):
        # TODO: but here we should consider to re-construct the get_residual() function
        r_u, r_v, r_div = self.DGS_obj.get_residual()
        ushape = r_u.shape
        vshape = r_v.shape
        pshape = r_div.shape

        start = timer()
        err = 1
        tol = 1e-6
        n_pre = 10  # the times of pre-smoothing

        while err > tol:
            du = np.zeros(ushape)
            dv = np.zeros(vshape)
            dp = np.zeros(pshape)









    def Vcycle_kernel(self):
        ff = 0





