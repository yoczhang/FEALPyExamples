#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: StokesHHORate2d.py
# @Author: Yongchao Zhang
# @E-mail: yoczhang@126.com
# @Time: May 20, 2020
# ---

# from fealpy.pde.stokes_model_2d import StokesModelData_0, StokesModelData_1, StokesModelData_2, StokesModelData_3
from Stokes2DData import StokesModelData_0
import numpy as np
from fealpy.mesh.Quadtree import Quadtree
import matplotlib.pyplot as plt
from fealpy.tools.show import showmultirate, show_error_table
from StokesHHOModel2d import StokesHHOModel2d
from fealpy.mesh.simple_mesh_generator import triangle
from fealpy.mesh.mesh_tools import find_entity


import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)

# --- begin setting --- #
d = 2  # the dimension
p = 1  # the polynomial order
n = 2  # the number of refine mesh
maxit = 4  # the max iteration of the mesh

pde = StokesModelData_0()  # create pde model

# # error settings
errorType = ['$|| u - u_h||_0$', '$||\\nabla u - \\nabla u_h||_0$', '|| p - p_h ||_0']
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)

Ndof = np.zeros(maxit, dtype=np.int)  # the array to store the number of dofs

# --- mesh setting --- #
# # mesh 1:
# # quad-tree mesh
# qtree = pde.init_mesh(n, meshtype='quad')
# mesh = qtree.to_pmesh()
node = np.array([
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1)], dtype=np.float)
cell = np.array([(0, 1, 2, 3)], dtype=np.int)
qtree = Quadtree(node, cell)
qtree.uniform_refine(n-2)
mesh = qtree.to_pmesh()

# # mesh 2:
# # tri mesh
# h = 1./4
# box = [0, 1, 0, 1]  # [0, 1]^2 domain
# mesh = triangle(box, h, meshtype='tri')

# # mesh 3:
# # polygon mesh
# h = 1./4
# box = [0, 1, 0, 1]  # [0, 1]^2 domain
# mesh = triangle(box, h, meshtype='polygon')

# --- plot the mesh --- #
# fig = plt.figure()
# axes = fig.gca()
# mesh.add_plot(axes, cellcolor='w')
# find_entity(axes, mesh, entity='cell', index=None, showindex=True, color='b', markersize=10, fontsize=8)
# find_entity(axes, mesh, entity='edge', index=None, showindex=True, color='r', markersize=10, fontsize=8)
# find_entity(axes, mesh, entity='node', index=None, showindex=True, color='y', markersize=10, fontsize=8)
# plt.show()
# plt.close()

pde.nu = 1.0
# --- start for-loop --- #
for i in range(maxit):
    stokes = StokesHHOModel2d(pde, mesh, p)
    s = stokes.solve()
    Ndof[i] = stokes.space.number_of_global_dofs()  # get the number of dofs
    errorMatrix[0, i] = stokes.velocity_L2_error()  # get the velocity L2 error
    errorMatrix[1, i] = stokes.velocity_energy_error()  # get the velocity energy error
    errorMatrix[2, i] = stokes.pressure_L2_error()  # get the pressure L2 error
    if i < maxit - 1:
        if mesh.meshtype == 'polygon':
            if 'qtree' in locals().keys():
                qtree.uniform_refine()  # uniform refine the mesh
                mesh = qtree.to_pmesh()  # transfer to polygon mesh
            else:
                h = h/2
                box = [0, 1, 0, 1]  # [0, 1]^2 domain
                mesh = triangle(box, h, meshtype='polygon')
        else:
            mesh.uniform_refine()


# --- get the convergence rate --- #
# # show the error table
show_error_table(Ndof, errorType, errorMatrix)

# # plot the rate
# showmultirate(plt, 0, Ndof, errorMatrix, errorType)
# plt.show()