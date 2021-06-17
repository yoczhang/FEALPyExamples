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
from FourierSpace_test import FourierSpace



# # Initial parameters settings
N = 2**7
h = 2*np.pi/N  # domain:[0,2*pi]^2
dt = 0.01
T = 1

epsilon, gamma, beta, alpha = 0.1, 0.01, 0.1, 1  # value of parameters
c = 3

# # Initial value u0 (t = 0)
uin = 0.05*(2*np.random.rand(N, N) - 1)
uaver = np.sum(uin)/N**2
u0 = uin - uaver

# # Solve the problem by SAV
ch = SAVCHModel()


