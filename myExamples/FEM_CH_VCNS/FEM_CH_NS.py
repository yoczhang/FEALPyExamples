#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: FEM_CH_NS.py
# @Author: Yongchao Zhang, Northwest University, Xi'an
# @E-mail: yoczhang@nwu.edu.cn
# @Site:
# @Time: Aug 10, 2021
# ---

__doc__ = """
The fealpy-FEM program for coupled Cahn-Hilliard-Navier-Stokes equation.
"""

import numpy as np
import matplotlib.pyplot as plt
from CahnHilliard2DData import CahnHilliardData0
from FEM_CH_NS_Model2d import FEM_CH_NS_Model2d
from fealpy.mesh import MeshFactory as MF
from PrintLogger import make_print_to_file
# from fealpy.tools.show import showmultirate, show_error_table
from to_show import show_error_table

# --- logging --- #
# make_print_to_file(filename='FEM_CH_NS', setpath="/Users/yczhang/Documents/FEALPy/FEALPyExamples/FEALPyExamples/myExamples/Logs/")

# --- begin setting --- #
d = 2  # the dimension
p = 4  # the polynomial order
n = 2  # the number of refine mesh
maxit = 5  # the max iteration of the mesh

t0 = 0.
T = 1
# NN = 64
box = [0, 1, 0, 1]
# mesh = MF.boxmesh2d(box, nx=NN, ny=NN, meshtype='tri')

start = 0  # (1/2)^0
stop = 4  # (1/2)^4
N_T = stop - start + 1
dt_space = 1e-1 * np.logspace(start, stop, N_T, base=1/2)
dt_min = min(dt_space)

time_scheme = 2  # 1 stands for 1st-order time-scheme; 2 is the 2nd-order time-scheme
h_space = dt_space ** (time_scheme/(p+0))

pdePars = {'m': 1e-3, 's': 1, 'alpha': 1, 'epsilon': 1e-3, 'eta': 1e-1, 'dt_min': dt_min, 'timeScheme': '1stOrder'}  # value of parameters
pde = CahnHilliardData0(t0, T)  # create pde model
pde.setPDEParameters(pdePars)

# # print some basic info
print('\n# ------------ the initial parameters ------------ #')
print('p = ', p)
print('t0 = %.4e' % t0)
print('dt_space = ', dt_space)
print('h_space = ', h_space)
print('domain box = ', box)
print('# #')

# # error settings
# errorType = ['$|| u - u_h||_0$', '$||\\nabla u - \\nabla u_h||_0$', '|| p - p_h ||_0']
errorType = ['$|| u - u_h||_0$', '$||\\nabla u - \\nabla u_h||_0$']
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)
Ndof = np.zeros(maxit, dtype=np.int)  # the array to store the number of dofs

# --- start for-loop --- #
for i in range(N_T):
    print('# ------------ in the time-mesh circle ------------ #')
    print('i = ', i)
    print('# -------------------------------------------------- #\n')
    NN = int(1./h_space[i]) + 1
    mesh = MF.boxmesh2d(box, nx=NN, ny=NN, meshtype='tri')
    ch = FEM_CH_NS_Model2d(pde, mesh, p, dt_space[i])
    if time_scheme == 1:
        l2err, h1err = ch.CH_NS_Solver_T1stOrder()
    else:
        l2err, h1err = ch.CH_NS_Solver_T1stOrder()

    Ndof[i] = ch.space.number_of_global_dofs()
    errorMatrix[0, i] = l2err  # get the velocity L2 error
    errorMatrix[1, i] = h1err  # get the velocity L2 error

# --- get the convergence rate --- #
print('# ------------ the error-table ------------ #')
show_error_table(dt_space, errorType, errorMatrix)

# # plot the rate
# showmultirate(plt, 0, Ndof, errorMatrix, errorType)
# plt.show()

print('# ------------ end of the file ------------ #')
