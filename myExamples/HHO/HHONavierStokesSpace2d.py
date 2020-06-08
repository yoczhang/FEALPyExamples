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
        """
        To solve the Navier-Stokes equation, we need the Newton iteration,
        this function is designed to get the matrices uesed in the Newton iteration.
        :param lastuh: last step solution uh,
        :return:
        """
        p = self.p
        mesh = self.mesh
        NC = mesh.number_of_cells()
        NE = mesh.number_of_edges()
        node = mesh.entity('node')
        edge = mesh.entity('edge')
        edge2cell = mesh.ds.edge_to_cell()
        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])

        hE = self.integralalg.edgemeasure  # (NE,), the length of edges
        n = mesh.edge_unit_normal()  # (NE,2), the unit normal vector of edges
        # # The direction of normal vector is from edge2cell[i,0] to edge2cell[i,1]
        # # (that is, from the cell with smaller number to the cell with larger number).

        qf = GaussLegendreQuadrature(p + 3)
        bcs, ws = qf.quadpts, qf.weights  # bcs.shape: (NQ,2); ws.shape: (NQ,)
        ps = np.einsum('ij, kjm->ikm', bcs, node[edge])  # (NQ,NE,2), NE is the number of edges

        # --- the last uh settings --- #
        vDof = self.dof.vDof
        vgdof = vDof.number_of_global_dofs()
        vcldof = vDof.number_of_cell_local_dof()
        veldof = p + 1  # number of edge local dof
        lastuh1 = lastuh[:vgdof]
        lastuh2 = lastuh[vgdof:]


    def basis(self, point, index=None, p=None):
        return self.vSpace.basis(point, index=index, p=p)

    def grad_basis(self, point, index=None, p=None):
        return self.vSpace.grad_basis(point, index=index, p=p)

    def value(self, uh, point, index=None):
        NC = self.mesh.number_of_cells()
        smldof = self.vSpace.smldof
        return self.vSpace.value(uh[:NC*smldof, ...], point, index=index)

    def grad_value(self, uh, point, index=None):
        NC = self.mesh.number_of_cells()
        smldof = self.vSpace.smldof
        return self.vSpace.grad_value(uh[:NC*smldof, ...], point, index=index)

    def edge_basis(self, point, index=None, p=None):
        return self.vSpace.edge_basis(point, index=index, p=p)

    def edge_value(self, uh, bcs):
        phi = self.edge_basis(bcs)
        edge2dof = self.dof.vDof.edge_to_dof()

        dim = len(uh.shape) - 1
        s0 = 'abcdefg'[:dim]
        s1 = '...ij, ij{}->...i{}'.format(s0, s0)
        val = np.einsum(s1, phi, uh[edge2dof])
        return val












