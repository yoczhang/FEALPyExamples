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


from sympy import symbols, sqrt, diff, sin, cos, atan2
import sympy.concrete.summations

x, y, nu, pi = symbols('x y nu pi')
r, lam, theta, omega = symbols('r lam theta omega')

# r = sqrt(x*x + y*y)
# r_x = diff(r, x)
# r_y = diff(r, y)
r_x = x/sqrt(x**2 + y**2)
r_y = y/sqrt(x**2 + y**2)

# theta = atan2(y, x)
# theta_x = diff(theta, x)
# theta_y = diff(theta, y)
theta_x = -y/r**2
theta_y = x/r**2

Psi = sin((1 + lam) * theta) * cos(lam * omega) / (1 + lam) - cos((1 + lam) * theta) - sin((1 - lam) * theta) * cos(
            lam * omega) / (1 - lam) + cos((1 - lam) * theta)

gPsi = diff(Psi, theta)
ggPsi = diff(gPsi, theta)
gggPsi = diff(ggPsi, theta)

u1 = r ** lam * ((1 + lam) * sin(theta) * Psi + cos(theta) * gPsi)
u2 = r ** lam * (sin(theta) * gPsi - (1 + lam) * cos(theta) * Psi)

u1_x = diff(u1, r)*r_x + diff(u1, theta)*theta_x
u1_y = diff(u1, r)*r_y + diff(u1, theta)*theta_y
u2_x = diff(u2, r)*r_x + diff(u2, theta)*theta_x
u2_y = diff(u2, r)*r_y + diff(u2, theta)*theta_y


print('gPsi = ', gPsi)
print('ggPsi = ', ggPsi)
print('gggPsi = ', gggPsi)
print('u1_x = ', u1_x)
print('u1_y = ', u1_y)
print('u2_x = ', u2_x)
print('u2_y = ', u2_y)



