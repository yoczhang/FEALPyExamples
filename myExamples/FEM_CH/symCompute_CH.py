#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: symCompute_CH.py
# @Author: Yongchao Zhang, Northwest University, Xi'an
# @E-mail: yoczhang@nwu.edu.cn
# @Site:
# @Time: Jul 30, 2021
# ---


from sympy import *
import numpy as np

t, x, y, nu, pi = symbols('t x y nu pi')
# --- ex0 --- #
u = cos(pi*x)*cos(pi*y)*sin(t)

ut = diff(u, t)
ux = diff(u, x)
uy = diff(u, y)
uxx = diff(ux, x)
uyy = diff(uy, y)
print('ut = ', ut)
print('ux = ', ux)
print('uxx = ', uxx)
print('uy = ', uy)
print('uyy = ', uyy)

