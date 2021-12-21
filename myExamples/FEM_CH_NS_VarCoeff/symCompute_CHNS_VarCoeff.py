#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: symCompute_CHNS_VarCoeff.py
# @Author: Yongchao Zhang
# @Institution: Northwest University, Xi'an, Shaanxi, China
# @E-mail: yoczhang@126.com, yoczhang@nwu.edu.cn
# @Site: 
# @Time: Dec 19, 2021
# ---


from sympy import symbols, cos, sin, diff, exp
import numpy as np

t, x, y, pi = symbols('t x y pi')
epsilon, eta, m = symbols('epsilon eta m')
rho0, rho1, nu0, nu1 = symbols('rho0 rho1 nu0 nu1')

# # ------------------- ex0 ------------------- # #
# # the CH equation:
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

# # the NS equation:
# pre settings:
J_0 = -(rho0 - rho1)/2 * m * c_x
J_1 = -(rho0 - rho1)/2 * m * c_y
rho = (rho0 + rho1)/2 + (rho0 - rho1)/2 * u
nu = (nu0 + nu1)/2 + (nu0 - nu1)/2 * u

vel0 = sin(pi*x)*cos(pi*y)*sin(t)
vel1 = -cos(pi*x)*sin(pi*y)*sin(t)

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

vel_stress00 = vel0_x
vel_stress01 = 0.5*(vel0_y + vel1_x)
vel_stress10 = 0.5*(vel0_y + vel1_x)
vel_stress11 = vel1_y

p = sin(pi*x)*sin(pi*y)*cos(t)
p_x = diff(p, x)
p_y = diff(p, y)

# # 1. coupled CH: source term
g_CH = u_t - m*laplace_c + diff(u*vel0, x) + diff(u*vel1, y)

# # 2. coupled NS: source term
g_NS_0 = rho * vel0_t + rho * (vel0*vel0_x + vel1*vel0_y) + (J_0*vel0_x + J_1*vel0_y) - (diff(nu*vel_stress00, x) + diff(nu*vel_stress01, y)) + p_x + u * c_x
g_NS_1 = rho * vel1_t + rho * (vel0*vel1_x + vel1*vel1_y) + (J_0*vel1_x + J_1*vel1_y) - (diff(nu*vel_stress10, x) + diff(nu*vel_stress11, y)) + p_y + u * c_y

print('u = ', u)
print('laplace_x = ', laplace_x)
print('laplace_y = ', laplace_y)
print('vel0_x = ', vel0_x)
print('vel0_y = ', vel0_y)
print('vel1_x = ', vel1_x)
print('vel1_y = ', vel1_y)
print('g_CH = ', g_CH)
print('g_NS_0 = ', g_NS_0)
print('g_NS_1 = ', g_NS_1)

print('end of the file')

