#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: symCls.py
# @Author: Yongchao Zhang, Northwest University, Xi'an
# @E-mail: yoczhang@nwu.edu.cn
# @Site:
# @Time: Oct 11, 2020
# ---


from sympy import *

x, y, nu = symbols('x y nu')
# --- ex1 --- #
# u1 = -exp(x)*(y*cos(y)+sin(y))
# u2 = exp(x)*y*sin(y)
# p = 2*exp(x)*sin(y) - (2*(1-exp(1))*(cos(1)-1))

# --- ex1 --- #
u1 = -0.5*cos(x)**2*cos(y)*sin(y)
u2 = 0.5*cos(y)**2*cos(x)*sin(x)
p = sin(x) - sin(y)


u1x = diff(u1, x)
u1xx = diff(u1x, x)
u1y = diff(u1, y)
u1yy = diff(u1y, y)
print('u1xx = ', u1xx)
print('u1yy = ', u1yy)

u2x = diff(u2, x)
u2xx = diff(u2x, x)
u2y = diff(u2, y)
u2yy = diff(u2y, y)
print('u2xx = ', u2xx)
print('u2yy = ', u2yy)

px = diff(p, x)
py = diff(p, y)
print('px = ', px)
print('py = ', py)

f1 = -nu*(u1xx + u1yy) + px
f2 = -nu*(u2xx + u2yy) + py

print('f1 = ', f1)
print('f2 = ', f2)

