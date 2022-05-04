#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: plotTrueSolution.py
# @Author: Yongchao Zhang
# @Institution: Northwest University, Xi'an, Shaanxi, China
# @E-mail: yoczhang@126.com, yoczhang@nwu.edu.cn
# @Site: 
# @Time: May 01, 2022
# ---


from sympy import symbols, tanh, lambdify
import numpy as np
import matplotlib  # 为了解决画图时采用 GUI (plt.show()) 的形式时, python3.8 崩溃的情况.
matplotlib.use("TkAgg")  # 为了解决画图时采用 GUI (plt.show()) 的形式时, python3.8 崩溃的情况.
import matplotlib.pyplot as plt


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
vel0_domain0 = vel_bar * C * (1 - delta**2 + nu_hat * (delta**2 - R**2)) / (delta**(n+3) * (nu_hat - 1) + 1)
vel0_domain1 = vel_bar * C * (1 - R**2) / (delta**(n+3) * (nu_hat - 1) + 1)

vel0_domain0_f = lambdify([y, K], vel0_domain0, "numpy")
vel0_domain1_f = lambdify([y, K], vel0_domain1, "numpy")

# |---
dx = 1./100
x_domain1_0 = np.arange(-1, -0.5, dx)
x_domain0 = np.arange(-0.5, 0.5, dx)
x_domain1_1 = np.arange(0.5, 1+dx, dx)

kk = -0.01
y_domain1_0 = vel0_domain1_f(x_domain1_0, kk)
y_domain0 = vel0_domain0_f(x_domain0, kk)
y_domain1_1 = vel0_domain1_f(x_domain1_1, kk)

xx = np.concatenate([np.concatenate([x_domain1_0, x_domain0]), x_domain1_1])
yy = np.concatenate([np.concatenate([y_domain1_0, y_domain0]), y_domain1_1])

# plt.figure()
# plt.plot(xx, yy)
# plt.xlabel("time")
# plt.ylabel("V")
# plt.savefig('./truesolution' + '.png')
# plt.close()

print('end of the plot-true-solution file')

