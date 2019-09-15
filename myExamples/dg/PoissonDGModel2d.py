import numpy as np

from fealpy.functionspace.lagrange_fem_space import LagrangeFiniteElementSpace
from fealpy.fem.integral_alg import IntegralAlg

from fealpy.boundarycondition import DirichletBC

from scipy.sparse.linalg import spsolve

from timeit import default_timer as timer


class PoissonFEMModel(object):
    def __init__(self, pde, mesh, p, q=3):
        self.space = LagrangeFiniteElementSpace(mesh, p, spacetype='D')
        self.mesh = self.space.mesh
        self.pde = pde
        self.uh = self.space.function()
        self.uI = self.space.interpolation(pde.solution)
        self.cellmeasure = mesh.entity_measure('cell')
        self.integrator = mesh.integrator(q)
        self.integralalg = IntegralAlg(
            self.integrator, self.mesh, self.cellmeasure)
