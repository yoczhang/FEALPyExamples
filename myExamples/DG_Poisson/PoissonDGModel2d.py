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

from DGSpace2d import DiscontinuousGalerkinSpace2d
from fealpy.fem.integral_alg import IntegralAlg

from fealpy.boundarycondition import DirichletBC

from scipy.sparse.linalg import spsolve

from timeit import default_timer as timer


class PoissonDGModel2d(object):
    def __init__(self, pde, mesh, p, q=3):
        self.space = DiscontinuousGalerkinSpace2d(mesh, p)
        self.mesh = self.space.mesh
        self.pde = pde
        self.uh = self.space.function()
        self.uI = self.space.interpolation(pde.solution)
        self.cellmeasure = mesh.entity_measure('cell')
        self.integrator = mesh.integrator(q)
        self.integralalg = IntegralAlg(
            self.integrator, self.mesh, self.cellmeasure)
