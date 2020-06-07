#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: HHONavierStokesSpace2d.py
# @Author: Yongchao Zhang
# @E-mail: yoczhang@126.com
# @Time: Jun 04, 2020
# ---


import numpy as np
from numpy.linalg import inv
# from fealpy.common import block, block_diag
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye, bmat

# from fealpy.functionspace.function import Function
from fealpy.quadrature import GaussLegendreQuadrature
# from fealpy.quadrature import PolygonMeshIntegralAlg
# from fealpy.functionspace.ScaledMonomialSpace2d import SMDof2d, ScaledMonomialSpace2d
from myScaledMonomialSpace2d import SMDof2d, ScaledMonomialSpace2d
from HHOStokesSpace2d import HHOStokesDof2d, HHOStokesSpace2d


class HHONavierStokesDof2d:
    def __init__(self, mesh, p):
        self.mesh = mesh
        self.p = p
        self.vDof = self.velocityDof()
        self.pDof = self.pressureDof()

    def velocityDof(self):
        # # note that, this Dof only has the scalar Dof
        return HHOStokesDof2d(self.mesh, self.p).velocityDof()

    def pressureDof(self):
        return HHOStokesDof2d(self.mesh, self.p).pressureDof()

    def number_of_global_dofs(self):
        return 2 * self.vDof.number_of_global_dofs() + self.pDof.number_of_global_dofs()


class HHONavierStokesSpace2d:
    def __init__(self, mesh, p):
        self.mesh = mesh
        self.p = p
        self.dof = HHONavierStokesDof2d(mesh, p)
        self.stokesspace = HHOStokesSpace2d(mesh, p)
        self.vSpace = self.stokesspace.vSpace
        self.pSpace = self.stokesspace.pSpace
        self.integralalg = self.vSpace.integralalg

    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def system_matrix(self, nu):
        A = self.stokesspace.velocity_matrix()  # (2*vgdof,2*vgdof)
        B = self.stokesspace.divergence_matrix()  # (pgdof,2*vgdof)
        P = self.stokesspace.pressure_correction()  # (1,2*vgdof+pgdof)

    def convective_matrix(self, lastuh):
        p = self.p












