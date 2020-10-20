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

x, y, nu, pi = symbols('x y nu pi')
# --- ex0 --- #
# u0 = sin(pi * x) * cos(pi * y)
# u1 = -cos(pi * x) * sin(pi * y)
# p = 1 / (y ** 2 + 1) - pi / 4

# --- ex1 --- #
# u0 = -exp(x)*(y*cos(y)+sin(y))
# u1 = exp(x)*y*sin(y)
# p = 2*exp(x)*sin(y) - (2*(1-exp(1))*(cos(1)-1))

# --- ex2 --- #
u0 = -0.5*cos(x)**2*cos(y)*sin(y)
u1 = 0.5*cos(y)**2*cos(x)*sin(x)
p = sin(x) - sin(y)


u0x = diff(u0, x)
u0xx = diff(u0x, x)
u0y = diff(u0, y)
u0yy = diff(u0y, y)
print('u0x = ', u0x)
print('u0y = ', u0y)
print('u0xx = ', u0xx)
print('u0yy = ', u0yy)

u1x = diff(u1, x)
u1xx = diff(u1x, x)
u1y = diff(u1, y)
u1yy = diff(u1y, y)
print('u1x = ', u1x)
print('u1y = ', u1y)
print('u1xx = ', u1xx)
print('u1yy = ', u1yy)

px = diff(p, x)
py = diff(p, y)
print('px = ', px)
print('py = ', py)

f0 = -nu*(u0xx + u0yy) + px
f1 = -nu*(u1xx + u1yy) + py

print('f0 = ', f0)
print('f1 = ', f1)
