#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: totest_temp4.py
# @Author: Yongchao Zhang, Northwest University, Xi'an
# @E-mail: yoczhang@nwu.edu.cn
# @Site:
# @Time: Jun 18, 2021
# ---


import numpy as np

aa = np.arange(3)
bb = 0.1 * np.arange(3)


spam = {'A':123 ,'B':345,'C':345 }
for k,v in spam.items():
    print(k,v)


def compute(*args):
    a = args[0]
    b = args[1]
    r1 = a + b
    r2 = np.array([a - b, a - b])

    r = np.array([a + b, a - b, a - b])
    return r


dic = {'a': 1, 'b': 2}

cc = compute(dic)

print(cc)
