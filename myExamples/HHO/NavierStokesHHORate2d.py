#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: NavierStokesHHORate2d.py
# @Author: Yongchao Zhang
# @E-mail: yoczhang@126.com
# @Time: May 29, 2020
# ---


from NavierStokes2DData import NavierStokes2DData_0
import numpy as np
import matplotlib.pyplot as plt
from fealpy.tools.show import showmultirate, show_error_table
from NavierStokesHHOModel2d import NavierStokesHHOModel2d
from fealpy.mesh.mesh_tools import find_entity


# --- begin setting --- #
d = 2  # the dimension
p = 1  # the polynomial order
n = 2  # the number of refine mesh
maxit = 4  # the max iteration of the mesh

nu = 1.0
pde = NavierStokes2DData_0(nu)  # create pde model
mesh = pde.init_mesh(n, meshtype='polygon')

# # error settings
errorType = ['$|| u - u_h||_0$', '$||\\nabla u - \\nabla u_h||_0$', '|| p - p_h ||_0']
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)

Ndof = np.zeros(maxit, dtype=np.int)  # the array to store the number of dofs

# --- plot the mesh --- #
# fig = plt.figure()
# axes = fig.gca()
# mesh.add_plot(axes, cellcolor='w')
# find_entity(axes, mesh, entity='cell', index=None, showindex=True, color='b', markersize=10, fontsize=8)
# find_entity(axes, mesh, entity='edge', index=None, showindex=True, color='r', markersize=10, fontsize=8)
# find_entity(axes, mesh, entity='node', index=None, showindex=True, color='y', markersize=10, fontsize=8)
# plt.show()
# plt.close()

# --- start for-loop --- #
for i in range(maxit):
    ns = NavierStokesHHOModel2d(pde, mesh, p)
    sol = ns.solve_by_Newton_iteration()
    Ndof[i] = ns.space.number_of_global_dofs()  # get the number of dofs
    errorMatrix[0, i] = ns.velocity_L2_error()  # get the velocity L2 error
    errorMatrix[1, i] = ns.velocity_energy_error()  # get the velocity energy error
    errorMatrix[2, i] = ns.pressure_L2_error()  # get the pressure L2 error
    if i < maxit - 1:
        n += 1
        mesh = pde.init_mesh(n, meshtype=mesh.meshtype)


# --- get the convergence rate --- #
# # show the error table
show_error_table(Ndof, errorType, errorMatrix)

# # plot the rate
# showmultirate(plt, 0, Ndof, errorMatrix, errorType)
# plt.show()



