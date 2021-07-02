#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: totest_meshgrid.py
# @Author: Yongchao Zhang, Northwest University, Xi'an
# @E-mail: yoczhang@nwu.edu.cn
# @Site:
# @Time: Jul 02, 2021
# ---


import numpy as np

c = np.zeros((2, 8))

c[0, :] = [0.,  4.,  8., 12., 16., -12., -8., -4.]
c[1, :] = [0.,  1.,  2.,  3.,  4., -3., -2., -1.]

r = np.split(c, [1, ])

zz = np.meshgrid(*r)

print('end of the file')


