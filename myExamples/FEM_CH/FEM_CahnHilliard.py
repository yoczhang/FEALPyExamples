#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: FEM_CahnHilliard.py
# @Author: Yongchao Zhang, Northwest University, Xi'an
# @E-mail: yoczhang@nwu.edu.cn
# @Site:
# @Time: Jul 30, 2021
# ---

__doc__ = """
The fealpy-FEM program for Cahn-Hilliard equation.
The ref: 2019 (JCP YangZhiguo) An unconditionally energy-stable scheme based on an implicit auxiliary energy variable for 
            incompressible two-phase flows with different densities involving only precomputable coefficient matrices.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
from CahnHilliard2DData import CahnHilliardData0
from fealpy.tools.show import showmultirate, show_error_table
from FEMCahnHilliardModel2d import FEMCahnHilliardModel2d
from fealpy.mesh import MeshFactory as MF
from PrintLogger import make_print_to_file

# --- logging --- #
make_print_to_file(filename='FEMVCNS', path="/Users/yczhang/Documents/FEALPy/FEALPyExamples/FEALPyExamples/myExamples/Logs/")

# --- begin setting --- #
d = 2  # the dimension
p = 2  # the polynomial order
n = 2  # the number of refine mesh
maxit = 1  # the max iteration of the mesh

t0 = 0.
T = 5.
dt = 1.0e-3
NN = 64

box = [0, 1, 0, 1]
mesh = MF.boxmesh2d(box, nx=NN, ny=NN, meshtype='tri')

m = 1
epsilon = 0.01
pde = CahnHilliardData0(t0, T)  # create pde model

pdePars = {'m': 1, 's': 1, 'alpha': 1, 'epsilon': 0.1, 'eta': 1, 'ConstantCoefficient': False}  # value of parameters
pde.setPDEParameters(pdePars)

# # print some basic info
print('t0 = %e' % t0)
print('dt = %e' % dt)
print('domain box = ', box)
print('mesh subdivision = ', NN)

# # error settings
errorType = ['$|| u - u_h||_0$', '$||\\nabla u - \\nabla u_h||_0$', '|| p - p_h ||_0']
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)

Ndof = np.zeros(maxit, dtype=np.int)  # the array to store the number of dofs

# --- start for-loop --- #
for i in range(maxit):
    ns = FEMCahnHilliardModel2d(pde, mesh, p, dt)
    ns.CH_Solver()
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