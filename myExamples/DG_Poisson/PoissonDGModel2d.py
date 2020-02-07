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
from fealpy.fem.integral_alg import IntegralAlg
from scipy.sparse.linalg import spsolve
from timeit import default_timer as timer


class PoissonDGModel2d(object):
    def __init__(self, pde, mesh, p, q=3):
        self.space = DGScalarSpace2d(mesh, p)
        self.mesh = self.space.mesh
        self.pde = pde
        self.uh = self.space.function()
        self.cellmeasure = mesh.entity_measure('cell')
        self.integrator = mesh.integrator(q)
        self.integralalg = IntegralAlg(
            self.integrator, self.mesh, self.cellmeasure)

    def get_left_matrix(self):
        space = self.space
        epsilon = self.pde.epsilon  # epsilon may take -1, 0, 1
        eta = self.pde.eta  # the penalty coefficient
        S = space.stiff_matrix()
        AJIn, JAIn, JJIn = space.interiorEdge_matrix()
        AJDir, JADir, JJDir = space.DirichletEdge_matrix()

        A = S - (AJIn + AJDir) + epsilon*(JAIn + JADir) + eta*(JJIn + JJDir)

        return A

    def get_right_vector(self):
        space = self.space
        f = self.pde.source
        gD = self.pde.gD
        fh = space.source_vector(f)
        JADir, JJDir = space.DirichletEdge_vector(gD)

        epsilon = self.pde.epsilon
        eta = self.pde.eta

        return fh + epsilon*JADir + eta*JJDir

    def solve(self):
        start = timer()
        A = self.get_left_matrix()
        b = self.get_right_vector()
        end = timer()
        self.A = A
        print("Construct linear system time:", end - start)

        start = timer()
        self.uh[:] = spsolve(A, b)
        end = timer()
        print("Solve time:", end - start)

        ls = {'A': A, 'b': b, 'solution': self.uh.copy()}

        return ls  # return the linear system

    def L2_error(self, u):
        uh = self.uh  # note that, here, type(uh) is the space.function variable

        def f(x, index):
            return (u(x, index) - uh.value(x, index))**2
        e = self.integralalg.integral(f, celltype=True)

        return np.sqrt(e.sum())

    def H1_semi_error(self, gu):
        uh = self.uh

        def f(x, index):
            return (gu(x, index) - uh.grad_value(x, index))**2
        e = self.integralalg.integral(f, celltype=True)

        return np.sqrt(e.sum())















