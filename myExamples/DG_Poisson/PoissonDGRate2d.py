#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: PoissonDGRate2d.py
# @Author: Yongchao Zhang
# @E-mail: yoczhang@126.com
# @Time: Jan 31, 2020
# ---


from fealpy.pde.poisson_2d import CosCosData as PDE
import numpy as np
import matplotlib.pyplot as plt
from fealpy.tools.show import showmultirate, show_error_table
from PoissonDGModel2d import PoissonDGModel2d
from fealpy.mesh.mesh_tools import find_entity

import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)
# sys.path.append("/Users/yczhang/Documents/FEALPy/FEALPyExamples/myExamples/DG_Poisson")


# --- begin setting --- #
d = 2  # the dimension
p = 1  # the polynomial order
n = 1  # the number of refine mesh
maxit = 3  # the max iteration of the mesh

pde = PDE()  # create pde model
pde.epsilon = -1  # setting the DG-scheme parameter
# # epsilon maybe take -1, 0, 1,
# # the corresponding DG-scheme is called symmetric interior penalty Galerkin (SIPG),
# # incomplete interior penalty Galerkin (IIPG) and nonsymmetric interior penalty Galerkin (NIPG)

pde.eta = 9  # setting the penalty parameter
# # eta may change corresponding the polynomial order 'p' and 'epsilon'

# # error settings
errorType = ['$|| u - u_h||_0$', '$||\\nabla u - \\nabla u_h||_0$']
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)

Ndof = np.zeros(maxit, dtype=np.int)  # the array to store the number of dofs

qtree = pde.init_mesh(n, meshtype='quadtree')
mesh = qtree.to_pmesh()
# ---- poly mesh ----
# h = 0.2
# box = [0, 1, 0, 1]  # [0, 1]^2 domain
# mesh = triangle(box, h, meshtype='polygon')
# -------------------
# # TODO: need to test the 'tri'-mesh and give more simple way to construct polygon mesh

# --- plot the mesh --- #
# fig = plt.figure()
# axes = fig.gca()
# mesh.add_plot(axes, cellcolor='w')
# find_entity(axes, mesh, entity='cell', index='all', showindex=True, color='b', markersize=10, fontsize=8)
# find_entity(axes, mesh, entity='edge', index='all', showindex=True, color='r', markersize=10, fontsize=8)
# find_entity(axes, mesh, entity='node', index='all', showindex=True, color='y', markersize=10, fontsize=8)
# plt.show()


# --- start for-loop --- #
for i in range(maxit):
    dg = PoissonDGModel2d(pde, mesh, p, q=p+2)
    ls = dg.solve()
    Ndof[i] = dg.space.number_of_global_dofs()  # get the number of dofs
    errorMatrix[0, i] = dg.L2_error()  # get the L2 error
    errorMatrix[1, i] = dg.H1_semi_error()  # get the H1-semi error
    if i < maxit - 1:
        qtree.uniform_refine()  # 一致加密网格
        mesh = qtree.to_pmesh()


