#!/anaconda3/envs/FEALPy/bin python3.7
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: MainStokesMAC.py
# @Author: Yongchao Zhang
# @E-mail: yoczhang@126.com
# @Time: Sep 11, 2019
# ---


import numpy as np
from fealpy.tools.show import show_error_table
from getPDEdata import getPDEBasicData
from Smoothers_2D import DGS_smoother
from solutionData import solutionStokesData

StokesData = solutionStokesData()


def test_DGS():
    maxit = 3
    errorType = ['time', '$|| res ||_{l_2}$']
    errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)
    for k in range(maxit):
        n = 2 ** (k+2)
        pde = getPDEBasicData(StokesData, n)
        tol = 1
        step = 0
        while tol > 1e-6:
            DGS_obj = DGS_smoother(pde)
            # uh, vh, ph = DGS_obj.smoother()
            r_u, r_v, r_div = DGS_obj.get_residual()
            tol = np.sqrt(np.mean(r_u ** 2)) + np.sqrt(np.mean(r_v ** 2)) + np.sqrt(np.mean(r_div ** 2))
            step = step + 1

            errorMatrix[0, k] = time  # compute time
            errorMatrix[0, k] = dof  # compute dof
            errorMatrix[2, k] = tol  # compute tol














print("test end")





