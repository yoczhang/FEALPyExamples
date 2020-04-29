#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: totest_temp.py
# @Author: Yongchao Zhang
# @E-mail: yoczhang@126.com
# @Time: Apr 28, 2020
# ---

import numpy as np

a1 = np.array((1, 2, 3))
a2 = np.array((4, 5, 6))
aa = [a1, a2]

bb = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]])


def sum_t(x):
    r = x[0] + x[1]
    return r


cc = list(map(sum_t, zip(aa, bb)))


# ------------------------------------------------- #
print("End of this test file")