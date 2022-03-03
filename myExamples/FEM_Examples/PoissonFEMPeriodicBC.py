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
from periodicBoundarySettings import periodicBoundarySettings
from to_show import show_error_table
from fealpy.tools.show import showmultirate

# solver
# from fealpy.solver import PETScSolver
from scipy.sparse.linalg import spsolve
import pyamg
from poisson_periodic_2d import CosCosData as PDE


# |--- set periodic function
def set_periodic_edge_func(mesh):
    """
    :return: idxPeriodicEdge0, 表示区域网格 `左侧` 的边
             idxPeriodicEdge1, 表示区域网格 `右侧` 的边
             idxNotPeriodicEdge, 表示区域网格 `上下两侧` 的边
    """
    idxBdEdge = mesh.ds.boundary_face_index()  # all the boundary edge index

    mid_coor = mesh.entity_barycenter('edge')  # (NE,2)
    bd_mid = mid_coor[idxBdEdge, :]

    isPeriodicEdge0 = np.abs(bd_mid[:, 0] - 0.0) < 1e-8
    isPeriodicEdge1 = np.abs(bd_mid[:, 0] - 1.0) < 1e-8
    notPeriodicEdge = ~(isPeriodicEdge0 + isPeriodicEdge1)
    idxPeriodicEdge0 = idxBdEdge[isPeriodicEdge0]  # (NE_Peri,)
    idxPeriodicEdge1 = idxBdEdge[isPeriodicEdge1]  # (NE_Peri,)

    # |--- 检验 idxPeriodicEdge0 与 idxPeriodicEdge1 是否是一一对应的
    y_0 = mid_coor[idxPeriodicEdge0, 1]
    y_1 = mid_coor[idxPeriodicEdge1, 1]
    if np.allclose(np.sort(y_0), np.sort(y_1)) is False:
        raise ValueError("`idxPeriodicEdge0` and `idxPeriodicEdge1` are not 1-to-1.")
    idxNotPeriodicEdge = idxBdEdge[notPeriodicEdge]
    return idxPeriodicEdge0, idxPeriodicEdge1, idxNotPeriodicEdge


p = 3
n = 3
maxit = 5
d = 2

pde = PDE()
mesh = pde.init_mesh(n=n)

errorType = ['$|| u - u_h||_{\Omega,0}$', '$||\\nabla u - \\nabla u_h||_{\Omega, 0}$']
errorMatrix = np.zeros((2, maxit), dtype=np.float)
NDof = np.zeros(maxit, dtype=np.float)

for i in range(maxit):
    print("The {}-th computation:".format(i))

    space = LagrangeFiniteElementSpace(mesh, p=p)

    pBS = periodicBoundarySettings(mesh, space.dof, set_periodic_edge_func)
    DirEdgeInd = pBS.idxNotPeriodicEdge
    periodicDof0, periodicDof1, dirDof = pBS.set_boundaryDofs()

    # |--- test
    # pdof = np.concatenate([periodicDof0, periodicDof1])
    # print('pdof = ', pdof)
    # bddof, = np.nonzero(space.boundary_dof())
    # print('diff_pdof = ', np.setdiff1d(bddof, pdof))

    NDof[i] = space.number_of_global_dofs()
    bc = DirichletBC(space, pde.dirichlet, threshold=DirEdgeInd)

    uh = space.function()
    A = space.stiff_matrix()

    F = space.source_vector(pde.source)

    A, F = bc.apply(A, F, uh)

    F, A = pBS.set_periodicAlgebraicSystem(periodicDof0, periodicDof1, F, lhsM=A)

    uh[:] = spsolve(A, F).reshape(-1)

    errorMatrix[0, i] = space.integralalg.L2_error(pde.solution, uh)
    errorMatrix[1, i] = space.integralalg.L2_error(pde.gradient, uh.grad_value)

    if i < maxit - 1:
        mesh.uniform_refine()
        print('|--- refine the mesh')


# --- get the convergence rate --- #
print('# ------------ the error-table ------------ #')
show_error_table(NDof, errorType, errorMatrix, table_scheme='h')