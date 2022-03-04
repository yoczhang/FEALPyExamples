#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: totest_try_except.py
# @Author: Yongchao Zhang, Northwest University, Xi'an
# @E-mail: yoczhang@nwu.edu.cn
# @Site:
# @Time: Mar 04, 2022
# ---

import numpy as np

try:
    aa = np.loadtxt('displace_byM.dat', dtype=np.float64)
    print(type(aa))
except IOError:
    print('no displace_byM.dat')

if 'aa' in vars().keys():
    print("vars have 'aa'")

print('end of the file')
