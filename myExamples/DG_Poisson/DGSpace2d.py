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
from fealpy.quadrature import GaussLegendreQuadrature
from scipy.sparse import coo_matrix, csr_matrix, spdiags
from fealpy.functionspace.function import Function
from fealpy.functionspace.ScaledMonomialSpace2d import SMDof2d, ScaledMonomialSpace2d


class DGDof2d(SMDof2d):
    def __init__(self, mesh, p=1):
        super(DGDof2d, self).__init__(mesh, p)

    def __str__(self):
        return "Discontinuous Galerkin Dofs!"


class DiscontinuousGalerkinSpace2d(ScaledMonomialSpace2d):
    def __init__(self, mesh, p=1):
        super(DiscontinuousGalerkinSpace2d, self).__init__(mesh, p)

    def __str__(self):
        return "Discontinuous Galerkin finite element space!"

    def jumpjump_matrix(self):
        p = self.p
        mesh = self.mesh
        node = mesh.entity('node')

        edge = mesh.entity('edge')
        edge2cell = mesh.ds.edge_to_cell()

        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])  # the bool vars, to get the inner edges

        NC = mesh.number_of_cells()

        qf = GaussLegendreQuadrature(p + 1)  # the integral points on edges (1D)
        bcs, ws = qf.quadpts, qf.weights  # bcs.shape: (NQ,2); ws.shape: (NQ,)
        ps = np.einsum('ij, kjm->ikm', bcs, node[edge])  # ps.shape: (NQ,NE,2), NE is the number of edges
        phi0 = self.basis(ps, index=edge2cell[:, 0])  # phi0.shape: (NQ,NE,ldof), lodf is the number of local DOFs
        # # phi0 is the value of the cell basis functions on the one-side of the corresponding edges.
        phi1 = self.basis(ps[:, isInEdge, :], index=edge2cell[isInEdge, 1])
        # # phi1 is the value of the cell basis functions on the other-side of the corresponding edges.



