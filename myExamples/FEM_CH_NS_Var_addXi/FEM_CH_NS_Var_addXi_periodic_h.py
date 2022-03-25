#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: FEM_CH_NS_Var_addXi_periodic_h.py
# @Author: Yongchao Zhang
# @Institution: Northwest University, Xi'an, Shaanxi, China
# @E-mail: yoczhang@126.com, yoczhang@nwu.edu.cn
# @Site: 
# @Time: Mar 25, 2022
# ---


__doc__ = """
The fealpy-FEM program for Variable-Coefficient coupled Cahn-Hilliard-Navier-Stokes equation, 
1. add the solver for \\xi.
2. check periodic boundary condition.
3. check h.
"""

import numpy as np
import sys
import datetime
import matplotlib.pyplot as plt
from PeriodicData import PeriodicData
from PeriodicModel2d import PeriodicModel2d
from fealpy.mesh import MeshFactory as MF
from PrintLogger import make_print_to_file
# from fealpy.tools.show import showmultirate, show_error_table
from to_show import show_error_table

# --- logging --- #
make_print_to_file(filename='FEM_CH_NS_Var_addXi_periodic_h', setpath="/Users/yczhang/Documents/FEALPy/FEALPyExamples/FEALPyExamples/myExamples/Logs/")

daytime = datetime.datetime.now().strftime('%Y%m%d')
hourtime = datetime.datetime.now().strftime("%H%M%S")
syslog = sys.platform + '-' + daytime + '-' + hourtime
print('\n# ---------------- the code is beginning --------------- #')
print('# ------------ FEM_CH_NS_Var_addXi_periodic ------------ #')
print('the OS and day-hour is: ', syslog)

# --- begin setting --- #
d = 2  # the dimension
p = 2  # the polynomial order
n = 2  # the number of refine mesh
maxit = 5  # the max iteration of the mesh

t0 = 0.
T = 0.05
box = [0, 1, 0, 1]
# mesh = MF.boxmesh2d(box, nx=NN, ny=NN, meshtype='tri')

dt_space = 1e-5
dt_min = dt_space

time_scheme = 1  # 1 stands for 1st-order time-scheme; 2 is the 2nd-order time-scheme
h_space = [1/4, 1/8, 1/16, 1/32, 1/64]
N_T = len(h_space)

pdePars = {'m': 1e-3, 's': 1, 'alpha': 1, 'epsilon': 1e-3, 'eta': 1e-1, 'dt_min': dt_min, 'timeScheme': '1stOrder'
           }  # value of parameters
VarCoeff = {'rho0': 1e-0, 'rho1': 3e-0, 'nu0': 1e-2, 'nu1': 2e-2}
pde = PeriodicData(t0, T)  # create pde model
pde.setPDEParameters(pdePars)
pde.setPDEParameters(VarCoeff)

# # print some basic info
print('\n# ------------ in FEM_CH_NS_Var_addXi_periodic code ------------ #')
print('# ------------ the initial parameters ------------ #')
print('p = ', p)
print('t0 = %.4e,  T = %.4e' % (t0, T))
print('dt_space = ', dt_space)
print('domain box = ', box)
print('rho0 = %.4e,  rho1 = %.4e' % (VarCoeff['rho0'], VarCoeff['rho1']))
print('nu0 = %.4e,  nu1 = %.4e, ' % (VarCoeff['nu0'], VarCoeff['nu1']))
print('# #')

# # --- error settings
errorType = ['$|| u - u_h||_0$', '$||\\nabla u - \\nabla u_h||_0$',
                                 '$|| vel - vel_h||_0$', '$||\\nabla vel - \\nabla vel_h||_0$', '|| p - p_h ||_0']
errorMatrix = np.zeros((len(errorType), N_T), dtype=np.float)
Ndof = np.zeros(N_T, dtype=np.int)  # the array to store the number of dofs

# --- start for-loop --- #
if hasattr(pde, 'box'):
    box = pde.box
    print('PDE has new domain box = ', box)

for i in range(N_T):
    print('\n# *********************************************************************** # \n')
    print('# ------------ in the time-mesh circle ------------ #')
    print('i = ', i)
    print('# -------------------------------------------------- #')
    # NN = 128
    NN = int((box[1] - box[0]) / h_space[i]) + 1
    print('    In new looping, NN = ', NN)
    mesh = MF.boxmesh2d(box, nx=NN, ny=NN, meshtype='tri')
    if time_scheme == 1:
        ch = PeriodicModel2d(pde, mesh, p, dt_space)
        uh_l2err, uh_h1err, vel_l2err, vel_h1err, ph_l2err = ch.CH_NS_addXi_Solver_T1stOrder()
    else:
        raise ValueError("There has no other time-scheme")

    Ndof[i] = ch.number_of_global_dofs()
    errorMatrix[0, i] = uh_l2err
    errorMatrix[1, i] = uh_h1err
    errorMatrix[2, i] = vel_l2err
    errorMatrix[3, i] = vel_h1err
    errorMatrix[4, i] = ph_l2err

# --- get the convergence rate --- #
print('# ------------ the error-table ------------ #')
show_error_table(Ndof, errorType, errorMatrix, table_scheme='h')

# # plot the rate
# showmultirate(plt, 0, Ndof, errorMatrix, errorType)
# plt.show()

print('# ------------ end of the file ------------ #')
