#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: FourierPDEdata.py
# @Author: Yongchao Zhang, Northwest University, Xi'an
# @E-mail: yoczhang@nwu.edu.cn
# @Site:
# @Time: Jun 17, 2021
# ---


__doc__ = """
The PDE datas for Fourier spectral methods.
"""

from functools import wraps
import numpy as np

def attachAttribute(func):
    @wraps(func)
    def wrapThefunc():
        

class FourierPDE:
    def __init__(self, N, dt, T, L=2*np.pi):
        self.N = N
        self.h = L/N
        self.dt = dt
        self.T = T

    def pdeParameters(self, epsilon, gamma, beta, alpha):
        self.epsilon = epsilon

