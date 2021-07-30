#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: FEMVC_NavierStokes_channel.py
# @Author: Yongchao Zhang, Northwest University, Xi'an
# @E-mail: yoczhang@nwu.edu.cn
# @Site:
# @Time: Jul 02, 2021
# ---


__doc__ = """
The fealpy-FEM program for Navier-Stokes problem by using the Velocity-Correction method. 
"""
import sys
sys.path.append('/Users/yczhang/Documents/FEALPy/fealpyHHOPersonal/fealpyHHOPersonal')
sys.path.append('/Users/yczhang/Documents/FEALPy/FEALPyExamples/FEALPyExamples/myExamples/Tools')

import numpy as np
import matplotlib.pyplot as plt
from NavierStokes2DData_channel import NavierStokes2DData_channel
from fealpy.tools.show import showmultirate, show_error_table
from FEMNavierStokesModel2d_channel import FEMNavierStokesModel2d_channel
from fealpy.mesh import MeshFactory as MF
from ShowCls import ShowCls
from PrintLogger import make_print_to_file

# --- logging --- #
# make_print_to_file(filename='FEMVCNS', path="/Users/yczhang/Documents/FEALPy/FEALPyExamples/FEALPyExamples/myExamples/Logs/")

# --- begin setting --- #
d = 2  # the dimension
p = 1  # the polynomial order
n = 2  # the number of refine mesh
maxit = 1  # the max iteration of the mesh

dt = 2.0e-2
T = 3
NN = 32

box = [0, 1, 0, 1]
mesh = MF.boxmesh2d(box, nx=NN, ny=NN, meshtype='tri')

nu = 1.0e-2
pde = NavierStokes2DData_channel(nu)  # create pde model

# # print some basic info
print('dt = %e' % dt)
print('nu = %e' % nu)
print('domain box = ', box)
print('mesh subdivision = ', NN)

# # error settings
errorType = ['$|| u - u_h||_0$', '$||\\nabla u - \\nabla u_h||_0$', '|| p - p_h ||_0']
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)

Ndof = np.zeros(maxit, dtype=np.int)  # the array to store the number of dofs

# sc = ShowCls(p, mesh)
# sc.showMesh(markNode=True, markEdge=True, markCell=True)

# --- start for-loop --- #
for i in range(maxit):
    ns = FEMNavierStokesModel2d_channel(pde, mesh, p, dt, T)
    # ns.set_outflow_edge()
    # ns.set_Dirichlet_edge()
    # ns.set_velocity_inflow_dof()
    uh0, uh1, ph = ns.NS_VC_Solver()


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

uh_ = ns.vspace.function()
uh_[:] = np.sqrt(uh0 ** 2 + uh1 ** 2)
ph_ = ns.pspace.function()
ph_[:] = ph



fig = plt.figure()
axes = fig.gca(projection='3d')
uh_.add_plot(axes, cmap='rainbow')

fig1 = plt.figure()
axes1 = fig1.gca(projection='3d')
ph_.add_plot(axes1, cmap='rainbow')

plt.show()

# --- get the convergence rate --- #
# # show the error table
# show_error_table(Ndof, errorType, errorMatrix)

# # plot the rate
# showmultirate(plt, 0, Ndof, errorMatrix, errorType)
# plt.show()