#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: PoissonFEMPeriodicBC.py
# @Author: Yongchao Zhang
# @Institution: Northwest University, Xi'an, Shaanxi, China
# @E-mail: yoczhang@126.com, yoczhang@nwu.edu.cn
# @Site: 
# @Time: Feb 11, 2022
# ---


import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fealpy.pde.poisson_2d import CosCosData
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import DirichletBC

from fealpy.tools.show import showmultirate

# solver
# from fealpy.solver import PETScSolver
from scipy.sparse.linalg import spsolve
import pyamg
from fealpy.pde.poisson_2d import CosCosData as PDE


p = 2
n = 1
maxit = 4
d = 2

pde = PDE()
mesh = pde.init_mesh(n=n)

errorType = ['$|| u - u_h||_{\Omega,0}$',
             '$||\\nabla u - \\nabla u_h||_{\Omega, 0}$'
             ]
errorMatrix = np.zeros((2, maxit), dtype=np.float)
NDof = np.zeros(maxit, dtype=np.float)

for i in range(maxit):
    print("The {}-th computation:".format(i))

    space = LagrangeFiniteElementSpace(mesh, p=p)
    NDof[i] = space.number_of_global_dofs()
    bc = DirichletBC(space, pde.dirichlet)

    uh = space.function()
    A = space.stiff_matrix()

    F = space.source_vector(pde.source)

    A, F = bc.apply(A, F, uh)

    if d == 2:
        uh[:] = spsolve(A, F).reshape(-1)

    errorMatrix[0, i] = space.integralalg.L2_error(pde.solution, uh)
    errorMatrix[1, i] = space.integralalg.L2_error(pde.gradient, uh.grad_value)

    if i < maxit - 1:
        mesh.uniform_refine()

