#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: StokesLshape_sym.py
# @Author: Yongchao Zhang, Northwest University, Xi'an
# @E-mail: yoczhang@nwu.edu.cn
# @Site:
# @Time: Nov 19, 2020
# ---


from sympy import *


x, y, nu, pi = symbols('x y nu pi')
lam, theta, omega = symbols('lam theta omega')

Psi = sin((1 + lam) * theta) * cos(lam * omega) / (1 + lam) - cos((1 + lam) * theta) - sin((1 - lam) * theta) * cos(
            lam * omega) / (1 - lam) + cos((1 - lam) * theta)

gPsi = diff(Psi, theta)
ggPsi = diff(gPsi, theta)
gggPsi = diff(ggPsi, theta)
print('gPsi = ', gPsi)
print('ggPsi = ', ggPsi)
print('gggPsi = ', gggPsi)

