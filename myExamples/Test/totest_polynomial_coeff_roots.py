#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: totest_polynomial_coeff_roots.py
# @Author: Yongchao Zhang
# @Institution: Northwest University, Xi'an, Shaanxi, China
# @E-mail: yoczhang@126.com, yoczhang@nwu.edu.cn
# @Site: 
# @Time: Feb 05, 2022
# ---


import numpy as np
import numpy.polynomial as poly

x = [1, 2, 3, 4, 5]
y = [16, 42.25, 81, 132.25, 196]

c = poly.Polynomial.fit(x, y, deg=2)
print(c(5))
print(c)
print(c.convert().coef)
print(c.roots())


# def ff(x):
#     val = 2.25 + 7.5 * x + 6.25 * x**2
#     return val








