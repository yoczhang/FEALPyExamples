#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: PoissonHHORate2d.py
# @Author: Yongchao Zhang
# @E-mail: yoczhang@126.com
# @Time: May 07, 2020
# ---


from fealpy.pde.poisson_2d import CosCosData as PDE
# from fealpy.pde.poisson_2d import SinSinData as PDE
import numpy as np
import matplotlib.pyplot as plt
from fealpy.tools.show import showmultirate, show_error_table
from PoissonHHOModel2d import PoissonHHOModel2d
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

pde = PDE()  # create pde model

# # error settings
errorType = ['$|| u - u_h||_0$', '$||\\nabla u - \\nabla u_h||_0$']
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)

Ndof = np.zeros(maxit, dtype=np.int)  # the array to store the number of dofs

# --- mesh setting --- #
# # mesh 1:
# # quad-tree mesh
qtree = pde.init_mesh(n-2, meshtype='quadtree')
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

# --- start for-loop --- #
for i in range(maxit):
    hho = PoissonHHOModel2d(pde, mesh, p)
    uh = hho.solve()
    # uh1 = hho.solve(solver='StaticCondensation')
    # uh2 = uh - uh1
    Ndof[i] = hho.smspace.number_of_global_dofs()  # get the number of dofs
    errorMatrix[0, i] = hho.L2_error()  # get the L2 error
    # errorMatrix[1, i] = hho.H1_semi_error()  # get the H1-semi error
    errorMatrix[1, i] = hho.energy_error()  # get the energy error
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

