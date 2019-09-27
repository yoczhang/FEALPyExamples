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
from timeit import default_timer as timer

from getPDEdata import getPDEBasicData
from Smoothers_2D import DGS_smoother
from solutionData import solutionStokesData
from showMACStokes import my_show_error_table

StokesData = solutionStokesData()


def DGS_test():
    maxit = 3
    ndof = np.zeros(maxit, dtype=np.int)
    errorType = ['time', 'iteStep', '$|| res ||_{l_2}$']
    errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)
    for k in range(maxit):
        n = 2 ** (k+3)
        pde = getPDEBasicData(StokesData, n)
        DGS_obj = DGS_smoother(pde)
        h = pde.h
        tol = 1
        step = 0
        start = timer()
        while tol > 1e-6:
            # ru, rv, rdiv = DGS_obj.get_residual()
            DGS_obj.smoother()
            r_u, r_v, r_div = DGS_obj.get_residual()
            tol = np.sqrt(np.mean(r_u ** 2)) + np.sqrt(np.mean(r_v ** 2)) + np.sqrt(np.mean(r_div ** 2))
            step = step + 1
        end = timer()

        ndof[k] = int(1/h)
        errorMatrix[0, k] = end - start  # compute time
        errorMatrix[1, k] = step  # compute time
        errorMatrix[2, k] = tol  # compute tol

    my_show_error_table(ndof, errorType, errorMatrix)

DGS_test()













print("test end")





