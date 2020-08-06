#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site:
# @File: PoissonDGRate2d.py
# @Author: Yongchao Zhang
# @E-mail: yoczhang@126.com
# @Time: Jan 31, 2020
# ---

import numpy as np

from DGScalarSpace2d import DGScalarSpace2d
from scipy.sparse.linalg import spsolve
from timeit import default_timer as timer


class PoissonDGModel2d(object):
    def __init__(self, pde, mesh, p):
        self.space = DGScalarSpace2d(mesh, p)
        self.mesh = self.space.mesh
        self.pde = pde
        self.uh = self.space.function()
        self.integralalg = self.space.integralalg

    def get_left_matrix(self):
        space = self.space
        epsilon = self.pde.epsilon  # epsilon may take -1, 0, 1
        eta = self.pde.eta  # the penalty coefficient
        isDirEdge = self.set_Dirichlet_edge()
        S = space.stiff_matrix()
        AJIn, JAIn, JJIn = space.interiorEdge_matrix()
        AJDir, JADir, JJDir = space.DirichletEdge_matrix(isDirEdge)

        A = S - (AJIn + AJDir) + epsilon*(JAIn + JADir) + eta*(JJIn + JJDir)

        return A

    def get_right_vector(self):
        space = self.space
        isDirEdge = self.set_Dirichlet_edge()
        f = self.pde.source
        uD = self.pde.dirichlet
        fh = space.source_vector(f)
        JADir, JJDir = space.DirichletEdge_vector(uD, isDirEdge)

        epsilon = self.pde.epsilon
        eta = self.pde.eta

        return fh + epsilon*JADir + eta*JJDir

    def solve(self):
        start = timer()
        A = self.get_left_matrix()
        b = self.get_right_vector()
        end = timer()
        print("Construct linear system time:", end - start)

        start = timer()
        self.uh[:] = spsolve(A, b)
        end = timer()
        print("Solve time:", end - start)

        ls = {'A': A, 'b': b, 'solution': self.uh.copy()}

        return ls  # return the linear system

    def L2_error(self):
        u = self.pde.solution
        uh = self.uh  # note that, here, type(uh) is the space.function variable

        def f(x, index):
            return (u(x) - uh.value(x, index))**2
        e = self.integralalg.integral(f, celltype=True)

        return np.sqrt(e.sum())

    def H1_semi_error(self):
        gu = self.pde.gradient
        uh = self.uh

        def f(x, index):
            return (gu(x) - uh.grad_value(x, index))**2
        e = self.integralalg.integral(f, celltype=True)

        return np.sqrt(e.sum())

    def set_Dirichlet_edge(self):
        mesh = self.mesh
        edge2cell = mesh.ds.edge_to_cell()
        bdEdge = (edge2cell[:, 0] == edge2cell[:, 1])  # the bool vars, to get the boundary edges

        isDirEdge = bdEdge  # here, we set all the boundary edges are Dir edges

        return isDirEdge















