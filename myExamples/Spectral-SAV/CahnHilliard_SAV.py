#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: CahnHilliard_SAV.py
# @Author: Yongchao Zhang, Northwest University, Xi'an
# @E-mail: yoczhang@nwu.edu.cn
# @Site:
# @Time: Jun 16, 2021
# ---

__doc__ = """
The Fourier spectral methods for Cahn-Hilliard equation by SAV.
The refer paper: 2019 (SIAM ShenJie) A New Class of Efficient and Robust Energy Stable Schemes for Gradient Flows
"""

import numpy as np
from fealpy.functionspace.FourierSpace import FourierSpace


