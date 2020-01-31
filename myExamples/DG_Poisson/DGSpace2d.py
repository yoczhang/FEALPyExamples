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

        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])

        NC = mesh.number_of_cells()

        qf = GaussLegendreQuadrature(p + 1)
        bcs, ws = qf.quadpts, qf.weights
        ps = np.einsum('ij, kjm->ikm', bcs, node[edge])
        phi0 = self.basis(ps, index=edge2cell[:, 0])
        phi1 = self.basis(ps[:, isInEdge, :], index=edge2cell[isInEdge, 1])

