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


from sympy import symbols, cos, sin, diff
import numpy as np

t, x, y, nu, pi = symbols('t x y nu pi')
epsilon, eta, m = symbols('epsilon eta m')

# --- ex0 --- #
u = cos(pi*x)*cos(pi*y)*sin(t)
h = epsilon/eta**2 * u * (u**2 - 1)

u_t = diff(u, t)
u_x = diff(u, x)
u_y = diff(u, y)
u_xx = diff(u_x, x)
u_yy = diff(u_y, y)

laplace_u = u_xx + u_yy
laplace_x = diff(laplace_u, x)
laplace_y = diff(laplace_u, y)

c = -epsilon*laplace_u + h
c_x = diff(c, x)
c_xx = diff(c_x, x)
c_y = diff(c, y)
c_yy = diff(c_y, y)
laplace_c = c_xx + c_yy

g = u_t - m*laplace_c

print('u = ', u)
print('laplace_x = ', laplace_x)
print('laplace_y = ', laplace_y)
print('g = ', g)

print('end of the file')


