#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: FEM_CH_NS_Var_addXi_CoCurrentFlow.py
# @Author: Yongchao Zhang
# @Institution: Northwest University, Xi'an, Shaanxi, China
# @E-mail: yoczhang@126.com, yoczhang@nwu.edu.cn
# @Site: 
# @Time: Feb 16, 2022
# ---

__doc__ = """
The fealpy-FEM program for Variable-Coefficient coupled Cahn-Hilliard-Navier-Stokes equation, 
add the solver for \\xi.
  |___ Co-current flow problem.
"""

import datetime
import numpy as np
import matplotlib.pyplot as plt
from CoCurrentFlowData import CoCurrentFlowTrueSolution
from CoCurrentFlowModel2d import CoCurrentFlowModel2d
from fealpy.mesh import MeshFactory as MF
from PrintLogger import make_print_to_file
# from fealpy.tools.show import showmultirate, show_error_table
from to_show import show_error_table

# --- logging --- #
# make_print_to_file(filename='FEM_CH_NS_Var_addXi_check_t', setpath="/Users/yczhang/Documents/FEALPy/FEALPyExamples/FEALPyExamples/myExamples/Logs/")

# --- begin setting --- #
d = 2  # the dimension
p = 2  # the polynomial order
n = 2  # the number of refine mesh

# |--- time and mesh setting
t0 = 0.
T = 0.1
dt_space = [1.e-5, ]
dt_min = min(dt_space)
time_scheme = 1  # 1 stands for 1st-order time-scheme; 2 is the 2nd-order time-scheme

# |--- pde setting
rho0 = 1.e-0
nu0 = 1.e-2
rho1 = 1.e-0
nu1 = nu0 * rho1 / rho0
sigma = 1.
eta = 1e-2
# eta = 8e-3
epsilon = 3./(2*np.sqrt(2))*sigma*eta

pdePars = {'m': 1e-5, 'epsilon': epsilon, 'eta': eta, 'dt_min': dt_min, 'timeScheme': '1stOrder'}  # value of parameters
varCoeff = {'rho0': rho0, 'rho1': rho1, 'nu0': nu0, 'nu1': nu1}
pde = CoCurrentFlowTrueSolution(t0, T, K=-0.01, r0=0.5, r1=1., nu0=nu0, nu1=nu1)  # create pde model
pde.setPDEParameters(pdePars)
pde.setPDEParameters(varCoeff)
mesh = pde.mesh

# # print some basic info
# |--- print some basic info
print('\n# ------------ in FEM_CH_NS_Var_addXi_CapillaryWave code ------------ #')
print('# ------------ the initial parameters ------------ #')
print('p = ', p)
print('t0 = %.4e,  T = %.4e' % (t0, T))
print('dt_space = ', dt_space)
print('domain box = ', pde.box)
print('rho0 = %.4e,  rho1 = %.4e' % (varCoeff['rho0'], varCoeff['rho1']))
print('nu0 = %.4e,  nu1 = %.4e, ' % (varCoeff['nu0'], varCoeff['nu1']))
print('m = %.4e, sigma = %.4e,  eta = %.4e, ' % (pdePars['m'], sigma, eta))
print('# #')

daytime = datetime.datetime.now().strftime('%Y%m%d')
hourtime = datetime.datetime.now().strftime("%H%M%S")

ccf = CoCurrentFlowModel2d(pde, mesh, p, dt_space[0])


print('end of the `FEM_CH_NS_Var_addXi_CocurrentFlow` code')
