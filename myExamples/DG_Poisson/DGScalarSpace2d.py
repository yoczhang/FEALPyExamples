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
from fealpy.functionspace.ScaledMonomialSpace2d import SMDof2d, ScaledMonomialSpace2d


class DGScalarDof2d(SMDof2d):
    def __init__(self, mesh, p=1):
        super(DGScalarDof2d, self).__init__(mesh, p)

    def __str__(self):
        return "Discontinuous Galerkin Dofs!"


class DGScalarSpace2d(ScaledMonomialSpace2d):
    def __init__(self, mesh, p=1):
        super(DGScalarSpace2d, self).__init__(mesh, p)

    def __str__(self):
        return "Discontinuous Galerkin finite element space!"

    def interiorEdge_matrix(self):  # get the average-jump, jump-average and jump-jump matrix at interior edges
        p = self.p
        mesh = self.mesh
        node = mesh.entity('node')

        edge = mesh.entity('edge')
        edge2cell = mesh.ds.edge_to_cell()
        edgeArea = mesh.edge_length()
        nm = mesh.edge_normal()
        # # (NE,2). The length of the normal-vector isn't 1, is the length of corresponding edge.

        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])  # the bool vars, to get the inner edges

        qf = GaussLegendreQuadrature(p + 1)  # the integral points on edges (1D)
        bcs, ws = qf.quadpts, qf.weights  # bcs.shape: (NQ,2); ws.shape: (NQ,)
        ps = np.einsum('ij, kjm->ikm', bcs, node[edge])  # ps.shape: (NQ,NE,2), NE is the number of edges

        phi0 = self.basis(ps[:, isInEdge, :], index=edge2cell[isInEdge, 0])
        # # phi0.shape: (NQ,NInE,ldof), NInE is the number of interior edges, lodf is the number of local DOFs
        # # phi0 is the value of the cell basis functions on the one-side of the corresponding edges.
        phi1 = self.basis(ps[:, isInEdge, :], index=edge2cell[isInEdge, 1])
        # # phi1 is the value of the cell basis functions on the other-side of the corresponding edges.

        gphi0 = self.grad_basis(ps[:, isInEdge, :], index=edge2cell[isInEdge, 0])
        # # gphi0.shape: (NQ,NInE,ldof,2), NInE is the number of interior edges, lodf is the number of local DOFs
        # # gphi0 is the grad-value of the cell basis functions on the one-side of the corresponding edges.
        gphi1 = self.grad_basis(ps[:, isInEdge, :], index=edge2cell[isInEdge, 1])

        # --- some explanations --- #
        # # In the following, the subscript 'm' stands for the smaller-index of the cell,
        # # and the subscript 'p' stands for the bigger-index of the cell.

        # --- get the average-jump matrix --- #
        AJmm = np.einsum('i, ijkm, ijp, jm->jpk', ws, gphi0, phi0, nm[isInEdge], optimize=True)  # (NInE,ldof,ldof)
        AJmp = np.einsum('i, ijkm, ijp, jm->jpk', ws, gphi0, phi1, nm[isInEdge], optimize=True)
        AJpm = np.einsum('i, ijkm, ijp, jm->jpk', ws, gphi1, phi0, nm[isInEdge], optimize=True)
        AJpp = np.einsum('i, ijkm, ijp, jm->jpk', ws, gphi1, phi1, nm[isInEdge], optimize=True)
        AJ_matrix = 0.5 * np.array([AJmm, -AJmp, AJpm, -AJpp])  # AJ_matrix.shape: (4,NInE,ldof,lodf)

        # --- get the jump-average matrix --- #
        JAmm = np.einsum('i, ijk, ijpm, jm->jpk', ws, phi0, gphi0, nm[isInEdge], optimize=True)  # (NInE,ldof,ldof)
        JAmp = np.einsum('i, ijk, ijpm, jm->jpk', ws, phi0, gphi1, nm[isInEdge], optimize=True)
        JApm = np.einsum('i, ijk, ijpm, jm->jpk', ws, phi1, gphi0, nm[isInEdge], optimize=True)
        JApp = np.einsum('i, ijk, ijpm, jm->jpk', ws, phi1, gphi1, nm[isInEdge], optimize=True)
        JA_matrix = 0.5 * np.array([JAmm, -JAmp, JApm, -JApp])  # JA_matrix.shape: (4,NInE,ldof,lodf)

        # --- get the jump-jump matrix --- #
        penalty = 1.0 / (edgeArea[isInEdge])
        JJmm = np.einsum('i, ijk, ijm, j, j->jmk', ws, phi0, phi0, edgeArea[isInEdge], penalty)  # Jmm.shape: (NInE,ldof,ldof)
        JJmp = np.einsum('i, ijk, ijm, j, j->jmk', ws, phi0, phi1, edgeArea[isInEdge], penalty)  # Jmp.shape: (NInE,ldof,ldof)
        JJpm = np.einsum('i, ijk, ijm, j, j->jmk', ws, phi1, phi0, edgeArea[isInEdge], penalty)  # Jpm.shape: (NInE,ldof,ldof)
        JJpp = np.einsum('i, ijk, ijm, j, j->jmk', ws, phi1, phi1, edgeArea[isInEdge], penalty)  # Jpp.shape: (NInE,ldof,ldof)
        JJ_matrix = np.array([JJmm, -JJmp, -JJpm, JJpp])  # JJ_matrix.shape: (4,NInE,ldof,lodf)

        # --- get the global dofs location --- #
        rowmm, colmm = self.global_dof_location(edge2cell[isInEdge, 0], edge2cell[isInEdge, 0])
        rowmp, colmp = self.global_dof_location(edge2cell[isInEdge, 0], edge2cell[isInEdge, 1])
        rowpm, colpm = self.global_dof_location(edge2cell[isInEdge, 1], edge2cell[isInEdge, 0])
        rowpp, colpp = self.global_dof_location(edge2cell[isInEdge, 1], edge2cell[isInEdge, 1])
        row = np.array([rowmm, rowmp, rowpm, rowpp])
        col = np.array([colmm, colmp, colpm, colpp])

        # --- construct the global matrix --- #
        gdof = self.number_of_global_dofs()
        AJ_matrix = csr_matrix((AJ_matrix.flat, (row.flat, col.flat)), shape=(gdof, gdof))
        JA_matrix = csr_matrix((JA_matrix.flat, (row.flat, col.flat)), shape=(gdof, gdof))
        JJ_matrix = csr_matrix((JJ_matrix.flat, (row.flat, col.flat)), shape=(gdof, gdof))

        return AJ_matrix, JA_matrix, JJ_matrix

    def DirichletEdge_matrix(self):  # get the average-jump, jump-average and jump-jump matrix at Dirichlet edges
        p = self.p
        mesh = self.mesh
        node = mesh.entity('node')

        edge = mesh.entity('edge')
        edge2cell = mesh.ds.edge_to_cell()
        edgeArea = mesh.edge_length()
        nm = mesh.edge_normal()
        # # (NE,2). The length of the normal-vector isn't 1, is the length of corresponding edge.

        isDirEdge = (edge2cell[:, 0] == edge2cell[:, 1])  # the bool vars, to get the inner edges

        qf = GaussLegendreQuadrature(p + 1)  # the integral points on edges (1D)
        bcs, ws = qf.quadpts, qf.weights  # bcs.shape: (NQ,2); ws.shape: (NQ,)
        ps = np.einsum('ij, kjm->ikm', bcs, node[edge])  # ps.shape: (NQ,NE,2), NE is the number of edges

        phi0 = self.basis(ps[:, isDirEdge, :], index=edge2cell[isDirEdge, 0])
        # # phi0.shape: (NQ,NDirE,ldof), NDirE is the number of Dirichlet edges, lodf is the number of local DOFs
        # # phi0 is the value of the cell basis functions on the one-side of the corresponding edges.

        gphi0 = self.grad_basis(ps[:, isDirEdge, :], index=edge2cell[isDirEdge, 0])
        # # gphi0.shape: (NQ,NDirE,ldof,2), NDirE is the number of Dirichlet edges, lodf is the number of local DOFs
        # # gphi0 is the grad-value of the cell basis functions on the one-side of the corresponding edges.

        # --- some explanations --- #
        # # In the following, the subscript 'm' stands for the smaller-index of the cell,
        # # and the subscript 'p' stands for the bigger-index of the cell.

        # --- get the average-jump matrix --- #
        AJmm = np.einsum('i, ijkm, ijp, jm->jpk', ws, gphi0, phi0, nm[isDirEdge], optimize=True)  # (NDirE,ldof,ldof)

        # --- get the jump-average matrix --- #
        JAmm = np.einsum('i, ijk, ijpm, jm->jpk', ws, phi0, gphi0, nm[isDirEdge], optimize=True)  # (NDirE,ldof,ldof)

        # --- get the jump-jump matrix --- #
        penalty = 1.0 / (edgeArea[isDirEdge])
        JJmm = np.einsum('i, ijk, ijm, j, j->jmk', ws, phi0, phi0, edgeArea[isDirEdge], penalty)  # Jmm.shape: (NDirE,ldof,ldof)

        # --- get the global dofs location --- #
        rowmm, colmm = self.global_dof_location(edge2cell[isDirEdge, 0], edge2cell[isDirEdge, 0])

        # --- construct the global matrix --- #
        gdof = self.number_of_global_dofs()
        AJmm = csr_matrix((AJmm.flat, (rowmm.flat, colmm.flat)), shape=(gdof, gdof))
        JAmm = csr_matrix((JAmm.flat, (rowmm.flat, colmm.flat)), shape=(gdof, gdof))
        JJmm = csr_matrix((JJmm.flat, (rowmm.flat, colmm.flat)), shape=(gdof, gdof))

        return AJmm, JAmm, JJmm

    def global_dof_location(self, trialCellIndex, testCellIndex):
        cell2dof = self.cell_to_dof()  # (NC,ldof)
        ldof = self.number_of_local_dofs()

        testdof = cell2dof[testCellIndex, :]  # (NtestCell,ldof)
        rowIndex = np.einsum('ij, k->ijk', testdof, np.ones(ldof))

        if trialCellIndex is not None:
            trialdof = cell2dof[trialCellIndex, :]  # (NtrialCell,ldof)
            # colIndex_temp = np.einsum('ij, k->ikj', trialdof, np.ones(ldof))
            # colIndex = colIndex_temp.swapaxes(-1, -2)
            colIndex = np.einsum('ij, k->ikj', trialdof, np.ones(ldof))
        else:
            colIndex = None

        return rowIndex, colIndex

    def stiff_matrix(self):
        p = self.p
        assert p >= 2, 'the polynomial-order should have p >= 2 '

        mesh = self.mesh
        node = mesh.entity('node')

        NC = mesh.number_of_cells()
        edge = mesh.entity('edge')
        edge2cell = mesh.ds.edge_to_cell()
        edgeArea = mesh.edge_length()
        nm = mesh.edge_normal()
        # # (NE,2). The length of the normal-vector isn't 1, is the length of corresponding edge.

        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])  # the bool vars, to get the inner edges

        qf = GaussLegendreQuadrature(p + 1)  # the integral points on edges (1D)
        bcs, ws = qf.quadpts, qf.weights  # bcs.shape: (NQ,2); ws.shape: (NQ,)
        ps = np.einsum('ij, kjm->ikm', bcs, node[edge])  # ps.shape: (NQ,NE,2), NE is the number of edges

        gphi0 = self.grad_basis(ps[:, isInEdge, :], index=edge2cell[:, 0])
        # # gphi0.shape: (NQ,NInE,ldof,2), NInE is the number of interior edges, lodf is the number of local DOFs
        # # gphi0 is the grad-value of the cell basis functions on the one-side of the corresponding edges.
        gphi1 = self.grad_basis(ps[:, isInEdge, :], index=edge2cell[isInEdge, 1])

        # # using the divergence-theorem to get the
        S0 = np.einsum('i, ijkm, ijpm->jpk', ws, gphi0, gphi0)  # (NE,ldof,ldof)
        b = node[edge[:, 0]] - self.cellbarycenter[edge2cell[:, 0]]  # (NE,2)
        S0 = np.einsum('ij, ij, ikm->ikm', b, nm, S0)  # (NE,ldof,ldof)

        S1 = np.einsum('i, ijkm, ijpm->jpk', ws, gphi1, gphi1)  # (NInE,ldof,ldof)
        b = node[edge[isInEdge, 0]] - self.cellbarycenter[edge2cell[isInEdge, 1]]  # (NInE,2)
        S1 = np.einsum('ij, ij, ikm->ikm', b, -nm[isInEdge], S1)  # (NInE,ldof,ldof)

        ldof = self.number_of_local_dofs()
        S = np.zeros((NC, ldof, ldof), dtype=np.float)
        np.add.at(S, edge2cell[:, 0], S0)
        np.add.at(S, edge2cell[isInEdge, 1], S1)

        multiIndex = self.dof.multiIndex
        q = np.sum(multiIndex, axis=1) - 1  # here, we used the grad-basis to get stiff-matrix, so we need to -1
        S /= q + q.reshape(-1, 1) + 2

        return S

    def source_vector(self, f):
        phi = self.basis  # basis is inherited from class ScaledMonomialSpace2d()

        def u(x, index):
            return np.einsum('ij, ijm->ijm', f(x), phi(x, index=index))
            # # f(x).shape: (NQ,NC).    phi(x,...).shape: (NQ,NC,ldof)

        fh = self.integralalg.integral(u, celltype=True)  # (NC,ldof)
        # # integralalg is inherited from class ScaledMonomialSpace2d()
        gdof = self.number_of_global_dofs()

        return fh.reshape(gdof,)

    def DirichletEdge_vector(self, gD):
        p = self.p
        mesh = self.mesh
        node = mesh.entity('node')

        edge = mesh.entity('edge')
        edge2cell = mesh.ds.edge_to_cell()
        edgeArea = mesh.edge_length()
        nm = mesh.edge_normal()
        # # (NE,2). The length of the normal-vector isn't 1, is the length of corresponding edge.

        isDirEdge = (edge2cell[:, 0] == edge2cell[:, 1])  # the bool vars, to get the inner edges

        qf = GaussLegendreQuadrature(p + 1)  # the integral points on edges (1D)
        bcs, ws = qf.quadpts, qf.weights  # bcs.shape: (NQ,2); ws.shape: (NQ,)
        ps = np.einsum('ij, kjm->ikm', bcs, node[edge])  # ps.shape: (NQ,NE,2), NE is the number of edges

        phi0 = self.basis(ps[:, isDirEdge, :], index=edge2cell[isDirEdge, 0])
        # # phi0.shape: (NQ,NDirE,ldof), NDirE is the number of Dirichlet edges, lodf is the number of local DOFs
        # # phi0 is the value of the cell basis functions on the one-side of the corresponding edges.

        gphi0 = self.grad_basis(ps[:, isDirEdge, :], index=edge2cell[isDirEdge, 0])
        # # gphi0.shape: (NQ,NDirE,ldof,2), NDirE is the number of Dirichlet edges, lodf is the number of local DOFs
        # # gphi0 is the grad-value of the cell basis functions on the one-side of the corresponding edges.

        gDh = gD(ps)
        # # (NQ,NE), get the Dirichlet values at physical integral points

        # --- get the jump-average, jump-jump vector at the Dirichlet bds --- #
        penalty = 1.0 / (edgeArea[isDirEdge])
        JADir_temp = np.einsum('i, ij, ijpm, jm->jp', ws, gDh, gphi0, nm[isDirEdge], optimize=True)  # (NDirE,ldof)
        JJDir_temp = np.einsum('i, ij, ijp, j, j->jp', ws, gDh, phi0, edgeArea[isDirEdge], penalty, optimize=True)  # (NDirE,ldof)

        # --- construct the final vector --- #
        NC = mesh.number_of_cells()
        shape = (NC,) + gDh.shape[2:]  # shape.shape: (NC,ldof)
        JADir = np.zeros(shape, dtype=np.float)
        JJDir = np.zeros(shape, dtype=np.float)

        np.add.at(JADir, edge2cell[isDirEdge, 0], JADir_temp)
        np.add.at(JJDir, edge2cell[isDirEdge, 0], JJDir_temp)














