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


# def attachAttributes(func):
#     @wraps(func)
#     def wrapThefunc(*args, **kwargs):
#         pars = args[1]
#         if pars is None:
#             return func(*args, **kwargs)
#         for k, v in pars.items():
#             setattr(func, k, v)
#         return func(*args, **kwargs)
#     return wrapThefunc
        

class FourierPDE:
    def __init__(self, N, dt, T, L=2*np.pi):
        self.N = N
        self.h = L/N
        self.dt = dt
        self.T = T

    def setPDEParameters(self, parameters):
        for k, v in parameters.items():
            self.__dict__[k] = v
        return None





