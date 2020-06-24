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
        self.stokesspace = self.space.stokesspace
        self.integralalg = self.space.integralalg
        self.uh0 = self.space.vSpace.function()
        self.uh1 = self.space.vSpace.function()
        self.ph = self.space.pSpace.function()
        self.A = None

    def solve_by_Newton_iteration(self):
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
        # aatest = AAS.copy()
        # bbtest = bbS.copy()
        lastuh = self.stokes_velocity_solver(AAS.copy(), bbS.copy())  # this solver will only get the solutions of uh0 and uh1
        tol = 1e-8
        err_it = 1.0
        Nit = 0
        pgdofp1 = self.space.pSpace.number_of_global_dofs() + 1

        start = timer()
        while (err_it > tol) & (Nit < 10):
            matrix1, matrix2, vec = self.space.convective_matrix(lastuh)
            convM = bmat([[matrix1 + matrix2, None],
                          [None, csr_matrix(np.zeros((pgdofp1, pgdofp1), dtype=self.ftype))]], format='csr')
            convV = np.concatenate([vec, np.zeros((pgdofp1, 1), dtype=self.ftype)], axis=0)
            AA = AAS + convM
            bb = bbS + convV
            self.A, b = self.applyDirichletBC(AA, bb)
            x = np.zeros(2*vgdof + pgdofp1,)
            x[:] = spsolve(self.A, b)
            uh0[:] = x[:vgdof]
            uh1[:] = x[vgdof:(2 * vgdof)]
            ph[:] = x[2*vgdof:-1]

            Nit += 1
            err_it = self.iteration_error(lastuh)
            lastuh = x[:2*vgdof]
            print("NS-iteration step: ", Nit)
            print("NS-iteration error: ", err_it)
        end = timer()
        print("NS-iteration solver time: ", end - start)

    def iteration_error(self, lastuh):
        vgdof = self.space.vSpace.number_of_global_dofs()
        lastuh0 = lastuh[:vgdof]
        lastuh1 = lastuh[vgdof:]

        scalarL2err = self.space.vSpace.L2_error

        err_it1 = scalarL2err(lastuh0, self.uh0)
        err_it2 = scalarL2err(lastuh1, self.uh1)
        return np.sqrt(err_it1*err_it1 + err_it2*err_it2)

    def stokes_velocity_solver(self, AA, bb):
        A, b = self.applyDirichletBC(AA, bb)
        uh0 = self.space.vSpace.function()
        uh1 = self.space.vSpace.function()
        ph = self.space.pSpace.function()
        vgdof = self.space.vSpace.number_of_global_dofs()

        # --- solve the system --- #
        x = np.concatenate([uh0, uh1, ph, np.zeros((1,), dtype=np.float)])  # (2*vgdof+pgdof+1,)
        start = timer()
        x[:] = spsolve(A, b)
        end = timer()
        print("Stokes-solver time: ", end - start)
        return x[:(2*vgdof)]

    def velocity_L2_error(self):
        velocity = self.pde.velocity
        uh0 = self.uh0
        uh1 = self.uh1
        return self.stokesspace.velocity_L2_error(velocity, uh0, uh1)

    def velocity_energy_error(self):
        velocity = self.pde.velocity
        uh0 = self.uh0
        uh1 = self.uh1
        A = self.A
        return self.stokesspace.velocity_energy_error(velocity, A, uh0, uh1)

    def pressure_L2_error(self):
        pressure = self.pde.pressure
        ph = self.ph
        return self.stokesspace.pressure_L2_error(pressure, ph)

    def applyDirichletBC(self, A, b):
        uD = self.pde.dirichlet  # uD(bcs): (NQ,NC,ldof,2)
        idxDirEdge = self.setDirichletEdges()
        AD, bD = self.space.stokesspace.applyDirichletBC(A, b, uD, idxDirEdge=idxDirEdge)
        return AD, bD

    def setDirichletEdges(self):
        # the following default Dirichlet edges
        return self.space.stokesspace.defaultDirichletEdges()

    # --- the following is for test functions
    def setFreeDofs(self):
        freedof = np.ones(self.space.dof.number_of_global_dofs() + 1, dtype=np.int)
        freedof[self.stokesspace.setStokesDirichletDofs()] = 0
        FreeDofs, = np.nonzero(freedof)
        return FreeDofs



