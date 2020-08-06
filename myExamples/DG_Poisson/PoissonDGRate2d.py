#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: PoissonDGModel2d.py
# @Author: Yongchao Zhang
# @Institution: Northwest University, Xi'an, Shaanxi, China
# @E-mail: yoczhang@126.com, yoczhang@nwu.edu.cn
# @Site:
# @Time: Jan 31, 2020
# ---


from fealpy.pde.poisson_2d import CosCosData as PDE
import numpy as np
import matplotlib.pyplot as plt
from fealpy.tools.show import showmultirate, show_error_table
from PoissonDGModel2d import PoissonDGModel2d
from fealpy.mesh.mesh_tools import find_entity
from fealpy.mesh import MeshFactory

# --- begin setting --- #
d = 2  # the dimension
p = 1  # the polynomial order
n = 2  # the number of refine mesh
maxit = 5  # the max iteration of the mesh

pde = PDE()  # create pde model
pde.epsilon = -1  # setting the DG-scheme parameter
# # epsilon maybe take -1, 0, 1,
# # the corresponding DG-scheme is called symmetric interior penalty Galerkin (SIPG),
# # incomplete interior penalty Galerkin (IIPG) and nonsymmetric interior penalty Galerkin (NIPG)

pde.eta = 16  # setting the penalty parameter
# # eta may change corresponding to the polynomial order 'p' and 'epsilon'

# # error settings
errorType = ['$|| u - u_h||_0$', '$||\\nabla u - \\nabla u_h||_0$']
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)

Ndof = np.zeros(maxit, dtype=np.int)  # the array to store the number of dofs

# --- mesh setting --- #
box = [0, 1, 0, 1]  # [0, 1]^2 domain
mf = MeshFactory()
meshtype = 'quad'
mesh = mf.boxmesh2d(box, nx=n, ny=n, meshtype=meshtype)

# --- plot the mesh --- #
fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes, cellcolor='w')
find_entity(axes, mesh, entity='cell', showindex=True, color='b', markersize=10, fontsize=8)
find_entity(axes, mesh, entity='edge', showindex=True, color='r', markersize=10, fontsize=8)
find_entity(axes, mesh, entity='node', showindex=True, color='y', markersize=10, fontsize=8)
plt.show()
plt.close()


# --- start for-loop --- #
for i in range(maxit):
    dg = PoissonDGModel2d(pde, mesh, p)
    ls = dg.solve()
    Ndof[i] = dg.smspace.number_of_global_dofs()  # get the number of dofs
    errorMatrix[0, i] = dg.L2_error()  # get the L2 error
    errorMatrix[1, i] = dg.H1_semi_error()  # get the H1-semi error
    if i < maxit - 1:
        mesh = mf.boxmesh2d(box, nx=n*2**(i+1), ny=n*2**(i+1), meshtype=meshtype)


# --- get the convergence rate --- #
# # show the error table
show_error_table(Ndof, errorType, errorMatrix)

# # plot the rate
showmultirate(plt, 0, Ndof, errorMatrix, errorType)
plt.show()
