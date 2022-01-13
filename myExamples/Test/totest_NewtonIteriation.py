#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: totest_NewtonIteriation.py
# @Author: Yongchao Zhang
# @Institution: Northwest University, Xi'an, Shaanxi, China
# @E-mail: yoczhang@126.com, yoczhang@nwu.edu.cn
# @Site: 
# @Time: Jan 13, 2022
# ---


from sympy import symbols, cos, sin, diff, exp
import numpy as np

# x = symbols('x')
# f = (x-1.2)**3 * (x-2)**2 * (x-3//2)
# fx = diff(f, x)
#
# print('f = ', f)
# print('fx = ', fx)
# print('end of the file')


def fx(x):
    return (x - 1.2) ** 3 * (x - 2) ** 2 * (x - 1.5)


def diff_fx(x):
    return 1.728*(0.833333333333333*x - 1)**3*(x - 2)**2 + 1.728*(0.833333333333333*x - 1)**3*(x - 1.5)*(2*x - 4) + 4.32*(0.833333333333333*x - 1)**2*(x - 2)**2*(x - 1.5)


def Newton_iteration(tol, x0):
    err = 1.
    while tol < err:
        xn = x0 - 1./diff_fx(x0) * fx(x0)
        err = np.abs(xn - x0)
        x0 = xn
        print('in Newton-iteration, err = ', err)
    return x0


if __name__ == '__main__':
    x0 = 1.
    tol = 1.e-10
    xn = Newton_iteration(tol, x0)

    print('end of the file')
