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

from fealpy.boundarycondition import DirichletBC

from scipy.sparse.linalg import spsolve

from timeit import default_timer as timer


class PoissonDGModel2d(object):
    def __init__(self, pde, mesh, p, q=3):
        self.space = DGScalarSpace2d(mesh, p)
        self.mesh = self.space.mesh
        self.pde = pde
        self.uh = self.space.function()
        self.uI = self.space.interpolation(pde.solution)
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
        fh = space.source_vector(f)



