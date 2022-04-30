#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: CoCurrentSymCompute.py
# @Author: Yongchao Zhang
# @Institution: Northwest University, Xi'an, Shaanxi, China
# @E-mail: yoczhang@126.com, yoczhang@nwu.edu.cn
# @Site: 
# @Time: Apr 23, 2022
# ---

from sympy import symbols, cos, sin, diff, exp, tanh
import numpy as np

K = symbols('K')
t, x, y, pi = symbols('t x y pi')
epsilon, m = symbols('epsilon m')
# rho0, rho1, nu0, nu1 = symbols('rho0 rho1 nu0 nu1')

rho0 = 1.
rho1 = 1.
nu0 = 0.01
nu1 = 0.06
r0 = 0.5
r1 = 1.
# eta = 0.0005
eta = 5.e-3

# 2D-par-setting
n = 0
R = y/r1
nu_hat = nu1/nu0
delta = r0/r1
C = (n + 3.)/2

# |--- CH
u = -tanh((y - r0)/(np.sqrt(2)*eta))

rho = (rho0 + rho1)/2 + (rho0 - rho1)/2 * u
nu = (nu0 + nu1)/2 + (nu0 - nu1)/2 * u

# |--- NS
vel_bar = K * r1 ** 2 / (nu1*(n+1)*(n+3)) * (delta**(n+3) * (nu_hat - 1) + 1)
vel0 = vel_bar * C * (1 - delta**2 + nu_hat * (delta**2 - R**2)) / (delta**(n+3) * (nu_hat - 1) + 1)
# vel0 = vel_bar * C * (1 - R**2) / (delta**(n+3) * (nu_hat - 1) + 1)
vel1 = 0
p = -K*x

vel0_t = diff(vel0, t)
vel0_x = diff(vel0, x)
vel0_xx = diff(vel0_x, x)
vel0_y = diff(vel0, y)
vel0_yy = diff(vel0_y, y)

vel1_t = diff(vel1, t)
vel1_x = diff(vel1, x)
vel1_xx = diff(vel1_x, x)
vel1_y = diff(vel1, y)
vel1_yy = diff(vel1_y, y)

vel_stress00 = 2*vel0_x
vel_stress01 = vel0_y + vel1_x
vel_stress10 = vel0_y + vel1_x
vel_stress11 = 2*vel1_y

p_x = diff(p, x)
p_y = diff(p, y)

g_NS_0 = rho * vel0_t + rho * (vel0*vel0_x + vel1*vel0_y) - (diff(nu*vel_stress00, x) + diff(nu*vel_stress01, y)) + p_x
g_NS_1 = rho * vel1_t + rho * (vel0*vel1_x + vel1*vel1_y) - (diff(nu*vel_stress10, x) + diff(nu*vel_stress11, y)) + p_y



print('end of the file')


