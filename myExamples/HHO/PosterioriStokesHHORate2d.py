#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: PosterioriStokesHHORate2d.py
# @Author: Yongchao Zhang, Northwest University, Xi'an
# @E-mail: yoczhang@nwu.edu.cn
# @Site:
# @Time: Sep 14, 2020
# ---

__doc__ = """
The fealpy program for posteriori Stokes problem. 
"""

from Stokes2DData import Stokes2DData_0
import numpy as np
from ShowCls import show
from StokesHHOModel2d import StokesHHOModel2d
from fealpy.mesh import MeshFactory
from fealpy.mesh import HalfEdgeMesh2d
from fealpy.tools.show import showmultirate, show_error_table
from fealpy.mesh.mesh_tools import find_entity
import matplotlib.pyplot as plt


# --- begin setting --- #
d = 2  # the dimension
p = 1  # the polynomial order
n = 2  # the number of refine mesh
maxit = 9  # the max iteration of the mesh

nu = 1.0e-3
pde = Stokes2DData_0(nu)  # create pde model

# --- error settings --- #
errorType = ['$|| u - u_h||_0$', '$||\\nabla u - \\nabla u_h||_0$', '|| p - p_h ||_0']
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)

Ndof = np.zeros(maxit, dtype=np.int)  # the array to store the number of dofs

# --- mesh setting --- #
box = [0, 1, 0, 1]  # [0, 1]^2 domain
mf = MeshFactory()
meshtype = 'quad'
mesh = mf.boxmesh2d(box, nx=n, ny=n, meshtype=meshtype)
mesh = HalfEdgeMesh2d.from_mesh(mesh)
mesh.init_level_info()
mesh.uniform_refine(n-1)  # refine the mesh at beginning

# --- plot the mesh --- #
# fig = plt.figure()
# axes = fig.gca()
# mesh.add_plot(axes, cellcolor='w')
# find_entity(axes, mesh, entity='cell', showindex=True, color='b', markersize=10, fontsize=8)
# find_entity(axes, mesh, entity='edge', showindex=True, color='r', markersize=10, fontsize=8)
# find_entity(axes, mesh, entity='node', showindex=True, color='y', markersize=10, fontsize=8)
# plt.show()

# --- start for-loop --- #
for i in range(maxit):
    stokes = StokesHHOModel2d(pde, mesh, p)
    sol = stokes.solve()
    Ndof[i] = stokes.space.number_of_global_dofs()  # get the number of dofs
    errorMatrix[0, i] = stokes.velocity_L2_error()  # get the velocity L2 error
    errorMatrix[1, i] = stokes.velocity_energy_error()  # get the velocity energy error
    errorMatrix[2, i] = stokes.pressure_L2_error()  # get the pressure L2 error

    # --- adaptive settings --- #
    uh = sol['uh']
    eta = stokes.space.residual_estimate0(nu, uh, pde.source)
    aopts = mesh.adaptive_options(method='max', theta=0.2, maxcoarsen=0, HB=True)
    print('before refine: number of cells: ', mesh.number_of_cells())
    mesh.adaptive(eta, aopts)
    # mesh.uniform_refine()
    print('after refine: number of cells: ', mesh.number_of_cells())


# --- get the convergence rate --- #
# # show the error table
show_error_table(Ndof, errorType, errorMatrix)

# # plot the rate
showmultirate(plt, 0, Ndof, errorMatrix, errorType)
plt.show()

# ---
print('end of the program')



