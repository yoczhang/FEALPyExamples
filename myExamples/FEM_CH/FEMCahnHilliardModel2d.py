#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: FEMCahnHilliardModel2d.py
# @Author: Yongchao Zhang, Northwest University, Xi'an
# @E-mail: yoczhang@nwu.edu.cn
# @Site:
# @Time: Aug 03, 2021
# ---


__doc__ = """
The FEM Cahn-Hilliard model in 2D. 
"""

import numpy as np
from scipy.sparse import csr_matrix, spdiags, eye, bmat
from fealpy.quadrature import FEMeshIntegralAlg
from scipy.sparse.linalg import spsolve
from fealpy.boundarycondition import DirichletBC
from fealpy.decorator import timer
from fealpy.functionspace import LagrangeFiniteElementSpace


class FEMCahnHilliardModel2d:
    def __init__(self, pde, mesh, p, dt):
        self.pde = pde
        self.p = p
        self.mesh = mesh
        self.timespace, self.dt = self.pde.time_mesh(dt)
        self.itype = self.mesh.itype
        self.ftype = self.mesh.ftype
        self.pde = pde
        self.space = LagrangeFiniteElementSpace(mesh, p)
        self.dof = self.space.dof
        self.cellmeasure = mesh.entity_measure('cell')
        self.integralalg = FEMeshIntegralAlg(self.mesh, p + 4, cellmeasure=self.cellmeasure)
        self.uh = self.space.function()
        self.wh = self.space.function()
        self.StiffMatrix = self.space.stiff_matrix()
        self.MassMatrix = self.space.mass_matrix()

    def setCoefficient(self, dt_minimum=None):
        pde = self.pde
        dt_min = self.dt/100 if dt_minimum is None else dt_minimum
        m = pde.m
        epslion = pde.epslion

        if pde.ConstantCoefficient is True:
            s = pde.s
            alpha = pde.alpha
            return s, alpha

        s = np.sqrt(4*epslion / (m * dt_min))
        alpha = 1./(2*epslion)*(-s + np.sqrt(s**2 - np.sqrt(4*epslion / (m * self.dt))))
        return s, alpha

    def uh_grad_value_at_faces(self, vh, f_bcs, cellidx, localidx):
        cell2dof = self.dof.cell2dof
        f_gphi = self.space.edge_grad_basis(f_bcs, cellidx, localidx)  # (NE,NQ,cldof,GD)
        val = np.einsum('ik, ijkm->jim', vh[cell2dof[cellidx]], f_gphi)  # (NQ,NE,GD)
        return val

    def grad_free_energy_cell(self, uh, c_bcs):
        space = self.space
        uh_val = space.value(uh, c_bcs)
        guh_val = space.grad_value(uh, c_bcs)


    @timer
    def CH_Solver(self):
        pde = self.pde
        dt = self.dt
        uh = self.uh
        wh = self.wh
        space = self.space
        dof = self.dof
        face2dof = dof.face_to_dof()  # (NE,fldof)
        cell2dof = dof.cell_to_dof()  # (NC,cldof)




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











