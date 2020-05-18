#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: PoissonHHOModel2d.py
# @Author: Yongchao Zhang
# @E-mail: yoczhang@126.com
# @Time: Feb 24, 2020
# ---

import numpy as np

from HHOScalarSpace2d import HHOScalarSpace2d
from HHOScalarSolver2d import HHOScalarSolver2d
from HHOBoundaryCondition import HHOBoundaryCondition
from numpy.linalg import inv
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from timeit import default_timer as timer


class PoissonHHOModel2d(object):
    def __init__(self, pde, mesh, p):
        self.p = p
        self.space = HHOScalarSpace2d(mesh, p)
        self.smspace = self.space.smspace
        self.mesh = self.space.mesh
        self.dof = self.space.dof
        self.pde = pde
        self.uh = self.space.function()
        self.uI = self.space.project(pde.solution)
        self.integralalg = self.space.integralalg
        self.A = None

    def get_left_matrix(self):
        space = self.space
        StiffM = space.reconstruction_stiff_matrix()  # list, its len is NC, each-term.shape (Cldof,Cldof)
        StabM = space.reconstruction_stabilizer_matrix()  # list, its len is NC, each-term.shape (Cldof,Cldof)

        def f(x): return x[0]+x[1]
        lM = list(map(f, zip(StiffM, StabM)))

        return lM  # list, its len is NC, each-term.shape (Cldof,Cldof)

    def get_right_vector(self):
        space = self.space
        f = self.pde.source
        RV = space.source_vector(f)  # (NC,ldof)

        return RV  # (NC,ldof)

    def solve(self, solver='direct'):
        uh = self.uh
        hhosolver = HHOScalarSolver2d(self, uh)

        start = timer()
        if solver == 'StaticCondensation':
            self.A, b = hhosolver.solving_by_static_condensation()
        elif solver == 'direct':
            self.A, b = hhosolver.solving_by_direct()
        end = timer()
        print("Solve time:", end - start)
        return uh

    # def L2_error(self):
    #     u = self.pde.solution
    #     uh = self.uh.value
    #     return self.space.integralalg.L2_error(u, uh)

    def L2_error(self):
        NC = self.mesh.number_of_cells()
        smldof = self.smspace.number_of_local_dofs()
        uI_cell = self.uI[:NC*smldof]  # (NC*smldof,)
        uh_cell = self.uh[:NC*smldof]  # (NC*smldof,)
        eu = (uI_cell - uh_cell)
        # eu = (uI_cell - uh_cell).reshape(NC, smldof)  # (NC,smldof)

        # def f(x, index):
        #     phi = self.space.basis(x, index=index)  # using the cell-integration, so phi: (NQ,NC,ldof)
        #     euphi = np.einsum('ijk, jk->ij', phi, eu)  # (NQ,NC)
        #     return euphi*euphi
        def f(x, index):
            evalue = self.space.value(eu, x, index=index)
            return evalue*evalue

        err = self.integralalg.integral(f)  # (NC,smldof)
        return np.sqrt(err)

    def H1_semi_error(self):
        gu = self.pde.gradient
        guh = self.uh.grad_value
        return self.space.integralalg.L2_error(gu, guh)

    def energy_error(self):
        e = self.uh - self.uI
        return np.sqrt(e@self.A@e)


















