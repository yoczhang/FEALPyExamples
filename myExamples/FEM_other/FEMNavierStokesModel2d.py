#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: FEMNavierStokesModel2d.py
# @Author: Yongchao Zhang, Northwest University, Xi'an
# @E-mail: yoczhang@nwu.edu.cn
# @Site:
# @Time: Jul 02, 2021
# ---


__doc__ = """
The FEM Navier-Stokes model in 2D. 
"""

import numpy as np
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.quadrature import FEMeshIntegralAlg


class FEMNavierStokesModel2d:
    def __init__(self, pde, mesh, p, dt, T):
        self.p = p
        self.mesh = mesh
        self.dt = dt
        self.T = T
        self.itype = self.mesh.itype
        self.ftype = self.mesh.ftype
        self.pde = pde
        self.vspace = LagrangeFiniteElementSpace(mesh, p+1)
        self.pspace = LagrangeFiniteElementSpace(mesh, p)
        self.cellmeasure = mesh.entity_measure('cell')
        self.integralalg = FEMeshIntegralAlg(self.mesh, p+4, cellmeasure=self.cellmeasure)
        self.uh0 = self.vspace.function()
        self.uh1 = self.vspace.function()
        self.ph = self.pspace.function()

    def solve_by_VCmethod(self):
        uh0 = self.uh0
        uh1 = self.uh1
        ph = self.ph

        vgdof = self.vspace.number_of_global_dofs()
        pgdof = self.pspace.number_of_global_dofs()

        # ---------------------------------------
        # get Stokes-system matrix
        # ---------------------------------------

    def NSNolinearTerm(self, uh0, uh1, bc):
        vspace = self.vspace





















