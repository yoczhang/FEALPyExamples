#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: FEMVC_NavierStokes_check_t.py
# @Author: Yongchao Zhang, Northwest University, Xi'an
# @E-mail: yoczhang@nwu.edu.cn
# @Site:
# @Time: Jul 02, 2021
# ---


__doc__ = """
The fealpy-FEM program for Navier-Stokes problem by using the Velocity-Correction method. 
"""

import numpy as np
import matplotlib.pyplot as plt
from NavierStokes2DData import NavierStokes2DData_time
from fealpy.tools.show import showmultirate, show_error_table
from FEMNavierStokesModel2d import FEMNavierStokesModel2d
from fealpy.mesh import MeshFactory as MF
# from PrintLogger import make_print_to_file

# --- logging --- #
# make_print_to_file(filename='FEMVCNS', path="/Users/yczhang/Documents/FEALPy/FEALPyExamples/FEALPyExamples/myExamples/Logs/")

# --- begin setting --- #
d = 2  # the dimension
p = 2  # the polynomial order
n = 2  # the number of refine mesh
maxit = 1  # the max iteration of the mesh

dt = 1.0e-1
T = 1
NN = 64

box = [0, 1, 0, 1]
mesh = MF.boxmesh2d(box, nx=NN, ny=NN, meshtype='tri')

start = 0  # (1/2)^0
stop = 4  # (1/2)^4
N_T = stop - start + 1
dt_space = 1e-1 * np.logspace(start, stop, N_T, base=1/2)

nu = 1.0e-2
pde = NavierStokes2DData_time(nu)  # create pde model

# # print some basic info
print('dt = %e' % dt)
print('nu = %e' % nu)
print('domain box = ', box)
print('mesh subdivision = ', NN)

# # error settings
errorType = ['$|| u - u_h||_0$', '$||\\nabla u - \\nabla u_h||_0$', '|| p - p_h ||_0']
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)

Ndof = np.zeros(maxit, dtype=np.int)  # the array to store the number of dofs

# --- start for-loop --- #
for i in range(maxit):
    ns = FEMNavierStokesModel2d(pde, mesh, p, dt, T)
    ns.NS_VC_Solver()
    # sol = ns.solve_by_Newton_iteration()
    # Ndof[i] = ns.space.number_of_global_dofs()  # get the number of dofs
    # errorMatrix[0, i] = ns.velocity_L2_error()  # get the velocity L2 error
    # errorMatrix[1, i] = ns.velocity_energy_error()  # get the velocity energy error
    # errorMatrix[2, i] = ns.pressure_L2_error()  # get the pressure L2 error
    # if i < maxit - 1:
    #     # qtree.uniform_refine()  # uniform refine the mesh
    #     # mesh = qtree.to_pmesh()  # transfer to polygon mesh
    #     n += 1
    #     mesh = pde.init_mesh(n, meshtype=mesh.meshtype)


# --- get the convergence rate --- #
# # show the error table
# show_error_table(Ndof, errorType, errorMatrix)

# # plot the rate
# showmultirate(plt, 0, Ndof, errorMatrix, errorType)
# plt.show()