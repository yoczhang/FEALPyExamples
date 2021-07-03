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
        pde = self.pde
        dt = self.dt
        uh0 = self.uh0
        uh1 = self.uh1
        ph = self.ph
        vspace = self.vspace
        pspace = self.pspace

        vgdof = self.vspace.number_of_global_dofs()
        pgdof = self.pspace.number_of_global_dofs()

        idxDirEdge = self.set_Dirichlet_edge()

        # init_uh0

        for nt in range(int(self.T/dt)):
            currt = nt * dt

            # ---------------------------------------
            # 1st-step: get the p^{n+1}
            # ---------------------------------------
            # # t^{n+1}: Pressure-Left-stiffMatrix
            plm = self.pspace.stiff_matrix()

            # # Pressure-Right-Matrix



    def NSNolinearTerm(self, uh0, uh1, bc):
        vspace = self.vspace
        val0 = vspace.value(uh0)  # val0.shape: (NQ,)
        val1 = vspace.value(uh1)
        gval0 = vspace.grad_value(uh0, bc)  # guh0.shape: (NQ,2)
        gval1 = vspace.grad_value(uh1, bc)

        NSNolinear = np.empty(gval0.shape, dtype=self.ftype)  # NSNolinear.shape: (NQ,2)

        NSNolinear[:, 0] = val0 * gval0[:, 0] + val1 * gval0[:, 1]
        NSNolinear[:, 1] = val0 * gval1[:, 0] + val1 * gval1[:, 1]

        return NSNolinear

    def set_Dirichlet_edge(self, idxDirEdge=None):
        if idxDirEdge is not None:
            return idxDirEdge

        mesh = self.mesh
        edge2cell = mesh.ds.edge_to_cell()
        isBdEdge = (edge2cell[:, 0] == edge2cell[:, 1])  # (NE,), the bool vars, to get the boundary edges

        isDirEdge = isBdEdge  # here, we set all the boundary edges are Dir edges
        idxDirEdge, = np.nonzero(isDirEdge)  # (NE_Dir,)

        return idxDirEdge

    def set_Neumann_edge(self, idxNeuEdge=None):
        if idxNeuEdge is not None:
            return idxNeuEdge

        mesh = self.mesh
        edge2cell = mesh.ds.edge_to_cell()
        bdEdge = (edge2cell[:, 0] == edge2cell[:, 1])  # the bool vars, to get the boundary edges

        isNeuEdge = bdEdge  # here, we first set all the boundary edges are Neu edges

        issetNeuEdge = 'no'
        if issetNeuEdge == 'no':
            isNeuEdge = None

        idxNeuEdge, = np.nonzero(isNeuEdge)  # (NE_Dir,)

        return idxNeuEdge























