#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: totest_temp3.py
# @Author: Yongchao Zhang, Northwest University, Xi'an
# @E-mail: yoczhang@nwu.edu.cn
# @Site:
# @Time: Oct 19, 2020
# ---

import numpy as np

aa = np.arange(12)
bb = 0.1*np.arange(12)


def compute(a, b):
    r1 = a + b
    r2 = np.array([a - b, a - b])

    r = np.array([a + b, a - b, a - b])
    return r


cc = np.array(list(map(compute, aa, bb)))

print(cc)

