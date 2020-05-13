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
        self.integralalg = self.space.integralalg

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
            hhosolver.solving_by_static_condensation()
        elif solver == 'direct':
            hhosolver.solving_by_direct()
        end = timer()
        print("Solve time:", end - start)
        return uh

    def L2_error(self):
        pass

    def H1_semi_error(self):
        pass

    def set_Dirichlet_edge(self):
        mesh = self.mesh
        edge2cell = mesh.ds.edge_to_cell()
        bdEdge = (edge2cell[:, 0] == edge2cell[:, 1])  # the bool vars, to get the boundary edges

        isDirEdge = bdEdge  # here, we set all the boundary edges are Dir edges

        return isDirEdge

















