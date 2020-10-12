#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: integrateCls.py
# @Author: Yongchao Zhang, Northwest University, Xi'an
# @E-mail: yoczhang@nwu.edu.cn
# @Site:
# @Time: Oct 11, 2020
# ---


__doc__ = """
这里将尝试构建 python 中用来积分的函数.
一重积分: integrate.quad
二重积分: integrate.dblquad
三重积分: integrate.tplquad
"""

import scipy.integrate as integrate
import numpy as np

exp = np.exp
sin = np.sin
cos = np.cos

# --- 例1: [0,1]x[0,1] 区间上的二重积分 --- #
# 这里注意积分区间的顺序
# 第二重积分的区间参数要以函数的形式传入
f = lambda x, y: 2 * exp(x) * sin(y) - 2 * (exp(1) - 1) * (1 - cos(1))
v, err = integrate.dblquad(f, 0, 1, lambda g: 0, lambda h: 1)  # 第二重积分的区间参数要以函数的形式传入
print('the integrate value:', v)

# --- 例2:  --- #
f = lambda x, y, z: x
g = lambda x: 0
h = lambda x: (1 - x) / 2
q = lambda x, y: 0
r = lambda x, y: 1 - x - 2 * y
v, err = integrate.tplquad(f, 0, 1, g, h, q, r)
print('the integrate value:', v)
