#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: SAVCahnHilliard.py
# @Author: Yongchao Zhang, Northwest University, Xi'an
# @E-mail: yoczhang@nwu.edu.cn
# @Site:
# @Time: Jun 16, 2021
# ---

__doc__ = """
The Fourier spectral methods for Cahn-Hilliard equation by SAV.
The reference paper: 2019 (SIAM ShenJie) A New Class of Efficient and Robust Energy Stable Schemes for Gradient Flows
"""

import numpy as np
from FourierPDEdata import FourierPDE
from SAVCHModel import SAVCHModel

# # Initial parameters settings
N = 2**2
h = 2*np.pi/N  # domain:[0,2*pi]^2
dt = 0.01
T = 1

pde = FourierPDE(N, dt, T)

pdePars = {'epsilon': 0.1, 'gamma': 0.01, 'beta': 0.1, 'alpha': 1}  # value of parameters
pde.setPDEParameters(pdePars)

# # Solve the problem by SAV
ch = SAVCHModel(pde)
ch.solve()


# ---
print('End of the file')

