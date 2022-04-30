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


# --- begin setting --- #
d = 2  # the dimension
p = 2  # the polynomial order
n = 2  # the number of refine mesh

# |--- time and mesh setting
t0 = 0.
T = 1
dt_space = [5.e-4, ]
dt_min = min(dt_space)
time_scheme = 1  # 1 stands for 1st-order time-scheme; 2 is the 2nd-order time-scheme

# |--- pde setting
rho0 = 1.e-0
rho1 = 1.e-0
nu0 = 1.e-2
nu1 = 6 * nu0
sigma = 0.
eta = 5e-3
epsilon = 3./(2*np.sqrt(2))*sigma*eta

# --- logging --- #
filename_basic = ('CCF_T(' + str(T) + ')_dt(' + ('%.e' % dt_space[0]) + ')_eta('
                  + ('%.e' % eta) + ')')
# make_print_to_file(filename=filename_basic,
#                    setpath="/Users/yczhang/Documents/FEALPy/FEALPyExamples/FEALPyExamples/myExamples/FEM_CH_NS_Var_addXi/CoCurrentFlowOutput/")

# |--- pde-parameters
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
val0_at_0 = ccf.CH_NS_addXi_Solver_T1stOrder()
# val0_at_0 = ccf.restart_CH_NS_addXi_Solver_T1stOrder('./CoCurrentFlowOutput/CCF_T(1)_dt(5e-04)_eta(5e-03)_nt(160)')

# filename = './CoCurrentFlowOutput/val0_at_0' + '_' + daytime + '-' + hourtime
# np.save(filename + '.npy', val0_at_0)
#
# plt.figure()
# plt.plot(val0_at_0[:, 0], val0_at_0[:, 1])
# plt.xlabel("time")
# plt.ylabel("V")
# plt.savefig(filename + '.png')
# plt.close()


print('end of the `FEM_CH_NS_Var_addXi_CocurrentFlow` code')
