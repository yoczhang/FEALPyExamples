#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: NavierStokesHHOModel2d.py
# @Author: Yongchao Zhang
# @E-mail: yoczhang@126.com
# @Time: Jun 14, 2020
# ---


import numpy as np
from HHONavierStokesSpace2d import HHONavierStokesSpace2d
from fealpy.quadrature import GaussLegendreQuadrature
from scipy.sparse import spdiags
# from numpy.linalg import inv
# from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from timeit import default_timer as timer
from scipy.sparse import csr_matrix, bmat


class NavierStokesHHOModel2d:
    def __init__(self, pde, mesh, p):
        self.p = p
        self.mesh = mesh
        self.itype = self.mesh.itype
        self.ftype = self.mesh.ftype
        self.pde = pde
        self.space = HHONavierStokesSpace2d(mesh, p)
        self.integralalg = self.space.integralalg
        self.uh0 = self.space.vSpace.function()
        self.uh1 = self.space.vSpace.function()
        self.ph = self.space.pSpace.function()
        self.A = None

    def solver_by_Newton_iteration(self, nu):
        # ---------------------------------------
        # get Stokes-system matrix
        # ---------------------------------------
        AAS = self.space.stokes_system_matrix(self.pde.nu)  # (2*vgdof+pgdof+1, 2*vgdof+pgdof+1)
        bbS = self.space.stokes_system_source(self.pde.source)

        # ---------------------------------------
        # get convective matrix
        # ---------------------------------------
        lastuh = self.space.vSpace.function()
        lastuh[:] = np.random.rand(len(lastuh))
        uh0 = np.concatenate([lastuh, lastuh])
        tol = 1e-8
        uherr = 1.0
        Nit = 0

        matrix1, matrix2, vec = self.space.convective_matrix(uh0)
        zerodof = AAS.shape[0] - matrix1.shape[0]

        while (uherr > tol) & (Nit < 30):
            convM = bmat([[matrix1 + matrix2, None], [None, csr_matrix(np.zeros((zerodof, zerodof), dtype=self.ftype))]])
            convV = np.concatenate([vec, np.zeros((zerodof, 1), dtype=self.ftype)], axis=0)
            AA = AAS + convM

    def applyDirichletBC(self, A, b):
        uD = self.pde.dirichlet  # uD(bcs): (NQ,NC,ldof,2)
        idxDirEdge = self.setDirichletEdges()
        AD, bD = self.space.stokesspace.applyDirichletBC(A, b, uD, idxDirEdge=idxDirEdge)
        return AD, bD

    def setDirichletEdges(self):
        # the following default Dirichelt edges
        return self.space.stokesspace.defaultDirichletEdges()

