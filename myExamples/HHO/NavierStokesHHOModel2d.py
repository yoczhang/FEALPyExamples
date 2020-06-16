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
        uh0 = self.uh0
        uh1 = self.uh1
        ph = self.ph
        vgdof = self.space.vSpace.number_of_global_dofs()

        # ---------------------------------------
        # get Stokes-system matrix
        # ---------------------------------------
        AAS = self.space.stokes_system_matrix(self.pde.nu)  # (2*vgdof+pgdof+1, 2*vgdof+pgdof+1)
        bbS = self.space.stokes_system_source(self.pde.source)

        # ---------------------------------------
        # get convective matrix
        # ---------------------------------------
        # lastuh = self.space.vSpace.function()
        # lastuh[:] = np.random.rand(len(lastuh))
        # lastuh = np.concatenate([lastuh, lastuh])
        lastuh = self.stokes_velocity_solver(AAS, bbS)[:-1]  # the number of all dofs is 2*vgdof+pgdof+1
        tol = 1e-8
        err_it = 1.0
        Nit = 0
        zerodof = self.space.pSpace.number_of_global_dofs() + 1

        while (err_it > tol) & (Nit < 30):
            matrix1, matrix2, vec = self.space.convective_matrix(lastuh)
            convM = bmat([[matrix1 + matrix2, None], [None, csr_matrix(np.zeros((zerodof, zerodof), dtype=self.ftype))]])
            convV = np.concatenate([vec, np.zeros((zerodof, 1), dtype=self.ftype)], axis=0)
            AA = AAS + convM
            bb = bbS + convV
            self.A, b = self.applyDirichletBC(AA, bb)
            x = np.concatenate([uh0, uh1, ph, np.zeros((1,), dtype=np.float)])  # (2*vgdof+pgdof+1,)
            x[:] = spsolve(self.A, b)
            uh0[:] = x[:vgdof]
            uh1[:] = x[vgdof:(2 * vgdof)]

            Nit += 1
            err_it = 1

    def iteration_error(self, lastuh):
        newuh = np.concatenate([self.uh0, self.uh1], axis=0)


    def stokes_velocity_solver(self, AA, bb):
        A, b = self.applyDirichletBC(AA, bb)
        uh0 = self.space.vSpace.function()
        uh1 = self.space.vSpace.function()
        ph = self.space.pSpace.function()
        vgdof = self.space.vSpace.number_of_global_dofs()

        # --- solve the system --- #
        x = np.concatenate([uh0, uh1, ph, np.zeros((1,), dtype=np.float)])  # (2*vgdof+pgdof+1,)
        start = timer()
        x[:] = spsolve(self.A, b)
        end = timer()
        print("Stokes-solver time:", end - start)

        uh0[:] = x[:vgdof]
        uh1[:] = x[vgdof:(2 * vgdof)]
        return np.concatenate([uh0, uh1], axis=0)

    def applyDirichletBC(self, A, b):
        uD = self.pde.dirichlet  # uD(bcs): (NQ,NC,ldof,2)
        idxDirEdge = self.setDirichletEdges()
        AD, bD = self.space.stokesspace.applyDirichletBC(A, b, uD, idxDirEdge=idxDirEdge)
        return AD, bD

    def setDirichletEdges(self):
        # the following default Dirichelt edges
        return self.space.stokesspace.defaultDirichletEdges()

