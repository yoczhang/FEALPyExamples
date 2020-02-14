#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site:
# @File: PoissonDGRate2d.py
# @Author: Yongchao Zhang
# @E-mail: yoczhang@126.com
# @Time: Feb 13, 2020
# ---

import numpy as np
from scipy.sparse.linalg import spsolve
from timeit import default_timer as timer
from fealpy.quadrature import GaussLegendreQuadrature
from scipy.sparse import csr_matrix
from fealpy.functionspace.ScaledMonomialSpace2d import ScaledMonomialSpace2d


class PoissonDGModel2d(object):
    def __init__(self, pde, mesh, p):
        self.smspace = ScaledMonomialSpace2d(mesh, p)
        self.mesh = mesh
        self.pde = pde
        self.p = p
        self.uh = self.smspace.function()
        self.integralalg = self.smspace.integralalg

    def get_left_matrix(self):
        epsilon = self.pde.epsilon  # epsilon may take -1, 0, 1
        eta = self.pde.eta  # the penalty coefficient
        isDirEdge = self.set_Dirichlet_edge()
        S = self.stiff_matrix()
        AJIn, JAIn, JJIn = self.interiorEdge_matrix()
        AJDir, JADir, JJDir = self.DirichletEdge_matrix(isDirEdge)

        A = S - (AJIn + AJDir) + epsilon*(JAIn + JADir) + eta*(JJIn + JJDir)

        return A

    def get_right_vector(self):
        isDirEdge = self.set_Dirichlet_edge()
        f = self.pde.source
        uD = self.pde.dirichlet
        fh = self.source_vector(f)
        JADir, JJDir = self.DirichletEdge_vector(uD, isDirEdge)

        epsilon = self.pde.epsilon
        eta = self.pde.eta

        return fh + epsilon*JADir + eta*JJDir

    def solve(self):
        start = timer()
        A = self.get_left_matrix()
        b = self.get_right_vector()
        end = timer()
        # self.A = A
        print("Construct linear system time:", end - start)

        start = timer()
        self.uh[:] = spsolve(A, b)
        end = timer()
        print("Solve time:", end - start)

        ls = {'A': A, 'b': b, 'solution': self.uh.copy()}

        return ls  # return the linear system

    def L2_error(self):
        u = self.pde.solution
        uh = self.uh  # note that, here, type(uh) is the space.function variable

        def f(x, index):
            return (u(x) - uh.value(x, index))**2
        e = self.integralalg.integral(f, celltype=True)

        return np.sqrt(e.sum())

    def H1_semi_error(self):
        gu = self.pde.gradient
        uh = self.uh

        def f(x, index):
            return (gu(x) - uh.grad_value(x, index))**2
        e = self.integralalg.integral(f, celltype=True)

        return np.sqrt(e.sum())

    def set_Dirichlet_edge(self):
        mesh = self.mesh
        edge2cell = mesh.ds.edge_to_cell()
        bdEdge = (edge2cell[:, 0] == edge2cell[:, 1])  # the bool vars, to get the boundary edges

        isDirEdge = bdEdge  # here, we set all the boundary edges are Dir edges

        return isDirEdge

    # --- Following functions are the kernel of the DG program --- #
    def interiorEdge_matrix(self):
        """
        Get the average-jump, jump-average and jump-jump matrix at interior edges.


        -------
        In the following, the subscript 'm'(-) stands for the smaller-index of the cell,
        and the subscript 'p'(+) stands for the bigger-index of the cell.

        What's more, let v, w be the trial and test function, respectively.

        Define
        (1) {{\nabla w}} := 1/2*(\nabla w^+ + \nabla w^-),
        (2) [[w]] := (w^- - w^+).
        (3) E_h: all the interior edges.
        (4) n_e: the unit-normal-vector of edge 'e'.
            In FEALPy, n_e is given by nm=mesh.edge_normal() (NE,2).
            Note that, the length of the normal-vector 'nm' isn't 1, is the length of corresponding edge.
            And the The direction of normal vector is from edge2cell[i,0] to edge2cell[i,1]
            (that is, from the cell with smaller number to the cell with larger number).


        -------
        The matrix
        AJ-matrix: \int_{E_h} {{\nabla v}}\cdot n_e [[w]],

        JA-matrix: \int_{E_h} [[v]]{{\nabla w}}\cdot n_e,

        JJ-matrix: \int_{E_h} [[v]][[w]].


        -------
        The DG scheme can be found in
        (Béatrice Rivière, Page:29) Discontinuous Galerkin Methods for Solving Elliptic and Parabolic Equations

        """

        smspace = self.smspace
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

        phi0 = smspace.basis(ps[:, isInEdge, :], index=edge2cell[isInEdge, 0])
        # # phi0.shape: (NQ,NInE,ldof), NInE is the number of interior edges, lodf is the number of local DOFs
        # # phi0 is the value of the cell basis functions on the one-side of the corresponding edges.
        phi1 = smspace.basis(ps[:, isInEdge, :], index=edge2cell[isInEdge, 1])
        # # phi1 is the value of the cell basis functions on the other-side of the corresponding edges.

        gphi0 = smspace.grad_basis(ps[:, isInEdge, :], index=edge2cell[isInEdge, 0])
        # # gphi0.shape: (NQ,NInE,ldof,2), NInE is the number of interior edges, lodf is the number of local DOFs
        # # gphi0 is the grad-value of the cell basis functions on the one-side of the corresponding edges.
        gphi1 = smspace.grad_basis(ps[:, isInEdge, :], index=edge2cell[isInEdge, 1])

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
        JA_matrix = 0.5 * np.array([JAmm, JAmp, -JApm, -JApp])  # JA_matrix.shape: (4,NInE,ldof,lodf)

        # --- get the jump-jump matrix --- #
        penalty = 1.0 / (edgeArea[isInEdge])
        JJmm = np.einsum('i, ijk, ijm, j, j->jmk', ws, phi0, phi0, edgeArea[isInEdge],
                         penalty)  # Jmm.shape: (NInE,ldof,ldof)
        JJmp = np.einsum('i, ijk, ijm, j, j->jmk', ws, phi0, phi1, edgeArea[isInEdge],
                         penalty)  # Jmp.shape: (NInE,ldof,ldof)
        JJpm = np.einsum('i, ijk, ijm, j, j->jmk', ws, phi1, phi0, edgeArea[isInEdge],
                         penalty)  # Jpm.shape: (NInE,ldof,ldof)
        JJpp = np.einsum('i, ijk, ijm, j, j->jmk', ws, phi1, phi1, edgeArea[isInEdge],
                         penalty)  # Jpp.shape: (NInE,ldof,ldof)
        JJ_matrix = np.array([JJmm, -JJmp, -JJpm, JJpp])  # JJ_matrix.shape: (4,NInE,ldof,lodf)

        # --- get the global dofs location --- #
        rowmm, colmm = self.global_dof_location(edge2cell[isInEdge, 0], edge2cell[isInEdge, 0])
        rowmp, colmp = self.global_dof_location(edge2cell[isInEdge, 0], edge2cell[isInEdge, 1])
        rowpm, colpm = self.global_dof_location(edge2cell[isInEdge, 1], edge2cell[isInEdge, 0])
        rowpp, colpp = self.global_dof_location(edge2cell[isInEdge, 1], edge2cell[isInEdge, 1])
        row = np.array([rowmm, rowmp, rowpm, rowpp])
        col = np.array([colmm, colmp, colpm, colpp])

        # --- construct the global matrix --- #
        gdof = smspace.number_of_global_dofs()
        AJ_matrix = csr_matrix((AJ_matrix.flat, (row.flat, col.flat)), shape=(gdof, gdof))
        JA_matrix = csr_matrix((JA_matrix.flat, (row.flat, col.flat)), shape=(gdof, gdof))
        JJ_matrix = csr_matrix((JJ_matrix.flat, (row.flat, col.flat)), shape=(gdof, gdof))

        return AJ_matrix, JA_matrix, JJ_matrix

    def DirichletEdge_matrix(self, isDirEdge):
        """
        Get the average-jump, jump-average and jump-jump matrix at Dirichlet edges.

        -------
        The explanations see interiorEdge_matrix()

        """

        smspace = self.smspace
        p = self.p
        mesh = self.mesh
        node = mesh.entity('node')

        edge = mesh.entity('edge')
        edge2cell = mesh.ds.edge_to_cell()
        edgeArea = mesh.edge_length()
        nm = mesh.edge_normal()
        # # (NE,2). The length of the normal-vector isn't 1, is the length of corresponding edge.

        # isDirEdge = (edge2cell[:, 0] == edge2cell[:, 1])  # the bool vars, to get the boundary edges

        qf = GaussLegendreQuadrature(p + 1)  # the integral points on edges (1D)
        bcs, ws = qf.quadpts, qf.weights  # bcs.shape: (NQ,2); ws.shape: (NQ,)
        ps = np.einsum('ij, kjm->ikm', bcs, node[edge])  # ps.shape: (NQ,NE,2), NE is the number of edges

        phi0 = smspace.basis(ps[:, isDirEdge, :], index=edge2cell[isDirEdge, 0])
        # # phi0.shape: (NQ,NDirE,ldof), NDirE is the number of Dirichlet edges, lodf is the number of local DOFs
        # # phi0 is the value of the cell basis functions on the one-side of the corresponding edges.

        gphi0 = smspace.grad_basis(ps[:, isDirEdge, :], index=edge2cell[isDirEdge, 0])
        # # gphi0.shape: (NQ,NDirE,ldof,2), NDirE is the number of Dirichlet edges, lodf is the number of local DOFs
        # # gphi0 is the grad-value of the cell basis functions on the one-side of the corresponding edges.

        # --- get the average-jump matrix --- #
        AJmm = np.einsum('i, ijkm, ijp, jm->jpk', ws, gphi0, phi0, nm[isDirEdge], optimize=True)  # (NDirE,ldof,ldof)

        # --- get the jump-average matrix --- #
        JAmm = np.einsum('i, ijk, ijpm, jm->jpk', ws, phi0, gphi0, nm[isDirEdge], optimize=True)  # (NDirE,ldof,ldof)

        # --- get the jump-jump matrix --- #
        penalty = 1.0 / (edgeArea[isDirEdge])
        JJmm = np.einsum('i, ijk, ijm, j, j->jmk', ws, phi0, phi0, edgeArea[isDirEdge],
                         penalty)  # Jmm.shape: (NDirE,ldof,ldof)

        # --- get the global dofs location --- #
        rowmm, colmm = self.global_dof_location(edge2cell[isDirEdge, 0], edge2cell[isDirEdge, 0])

        # --- construct the global matrix --- #
        gdof = smspace.number_of_global_dofs()
        AJmm = csr_matrix((AJmm.flat, (rowmm.flat, colmm.flat)), shape=(gdof, gdof))
        JAmm = csr_matrix((JAmm.flat, (rowmm.flat, colmm.flat)), shape=(gdof, gdof))
        JJmm = csr_matrix((JJmm.flat, (rowmm.flat, colmm.flat)), shape=(gdof, gdof))

        return AJmm, JAmm, JJmm

    def global_dof_location(self, trialCellIndex, testCellIndex):
        smspace = self.smspace

        cell2dof = smspace.cell_to_dof()  # (NC,ldof)
        ldof = smspace.number_of_local_dofs()

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
        """
        Get the stiff matrix on ScaledMonomialSpace2d.

        ---
        The mass matrix on ScaledMonomialSpace2d can be found in class ScaledMonomialSpace2d(): mass_matrix()

        """

        smspace = self.smspace
        p = self.p
        assert p >= 1, 'the polynomial-order should have p >= 1 '

        mesh = self.mesh
        node = mesh.entity('node')

        NC = mesh.number_of_cells()
        edge = mesh.entity('edge')
        edge2cell = mesh.ds.edge_to_cell()
        nm = mesh.edge_normal()
        # # (NE,2). The length of the normal-vector isn't 1, is the length of corresponding edge.

        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])  # the bool vars, to get the inner edges

        qf = GaussLegendreQuadrature(p + 1)  # the integral points on edges (1D)
        bcs, ws = qf.quadpts, qf.weights  # bcs.shape: (NQ,2); ws.shape: (NQ,)
        ps = np.einsum('ij, kjm->ikm', bcs, node[edge])  # ps.shape: (NQ,NE,2), NE is the number of edges

        gphi0 = smspace.grad_basis(ps, index=edge2cell[:, 0])
        # # gphi0.shape: (NQ,NInE,ldof,2), NInE is the number of interior edges, lodf is the number of local DOFs
        # # gphi0 is the grad-value of the cell basis functions on the one-side of the corresponding edges.
        gphi1 = smspace.grad_basis(ps[:, isInEdge, :], index=edge2cell[isInEdge, 1])

        # # using the divergence-theorem to get the
        S0 = np.einsum('i, ijkm, ijpm->jpk', ws, gphi0, gphi0)  # (NE,ldof,ldof)
        b = node[edge[:, 0]] - smspace.cellbarycenter[edge2cell[:, 0]]  # (NE,2)
        S0 = np.einsum('ij, ij, ikm->ikm', b, nm, S0)  # (NE,ldof,ldof)

        S1 = np.einsum('i, ijkm, ijpm->jpk', ws, gphi1, gphi1)  # (NInE,ldof,ldof)
        b = node[edge[isInEdge, 0]] - smspace.cellbarycenter[edge2cell[isInEdge, 1]]  # (NInE,2)
        S1 = np.einsum('ij, ij, ikm->ikm', b, -nm[isInEdge], S1)  # (NInE,ldof,ldof)

        ldof = smspace.number_of_local_dofs()
        S = np.zeros((NC, ldof, ldof), dtype=np.float)
        np.add.at(S, edge2cell[:, 0], S0)
        np.add.at(S, edge2cell[isInEdge, 1], S1)

        multiIndex = smspace.dof.multiIndex
        q = np.sum(multiIndex, axis=1) - 1  # here, we used the grad-basis to get stiff-matrix, so we need to -1
        qq = q + q.reshape(-1, 1) + 2
        qq[0, 0] = 1
        # # note that this is the special case, since we want to compute the \int_T \nabla u\cdot \nabla v,
        # # this needs to minus 1 in the 'q', so qq[0,0] is 0, moreover, S[:, 0, :] == S[:, :, 0] is 0-values,
        # # so we set qq[0, 0] = 1 which doesn't affect the result of S /= qq.

        S /= qq

        # --- get row and col --- #
        row, col = self.global_dof_location(np.arange(NC), np.arange(NC))

        gdof = smspace.number_of_global_dofs()
        S = csr_matrix((S.flat, (row.flat, col.flat)), shape=(gdof, gdof))

        return S

    def source_vector(self, f):
        smspace = self.smspace
        phi = smspace.basis

        def u(x, index):
            return np.einsum('ij, ijm->ijm', f(x), phi(x, index=index))
            # # f(x).shape: (NQ,NC).    phi(x,...).shape: (NQ,NC,ldof)

        fh = self.integralalg.integral(u, celltype=True)  # (NC,ldof)

        gdof = smspace.number_of_global_dofs()

        return fh.reshape(gdof, )

    def DirichletEdge_vector(self, uD, isDirEdge):
        smspace = self.smspace
        p = self.p
        mesh = self.mesh
        node = mesh.entity('node')

        edge = mesh.entity('edge')
        edge2cell = mesh.ds.edge_to_cell()
        edgeArea = mesh.edge_length()
        nm = mesh.edge_normal()
        # # (NE,2). The length of the normal-vector isn't 1, is the length of corresponding edge.

        # isDirEdge = (edge2cell[:, 0] == edge2cell[:, 1])  # the bool vars, to get the boundary edges

        qf = GaussLegendreQuadrature(p + 1)  # the integral points on edges (1D)
        bcs, ws = qf.quadpts, qf.weights  # bcs.shape: (NQ,2); ws.shape: (NQ,)
        ps = np.einsum('ij, kjm->ikm', bcs, node[edge])  # ps.shape: (NQ,NE,2), NE is the number of edges

        phi0 = smspace.basis(ps[:, isDirEdge, :], index=edge2cell[isDirEdge, 0])
        # # phi0.shape: (NQ,NDirE,ldof), NDirE is the number of Dirichlet edges, lodf is the number of local DOFs
        # # phi0 is the value of the cell basis functions on the one-side of the corresponding edges.

        gphi0 = smspace.grad_basis(ps[:, isDirEdge, :], index=edge2cell[isDirEdge, 0])
        # # gphi0.shape: (NQ,NDirE,ldof,2), NDirE is the number of Dirichlet edges, lodf is the number of local DOFs
        # # gphi0 is the grad-value of the cell basis functions on the one-side of the corresponding edges.

        uDh = uD(ps[:, isDirEdge, :])
        # # (NQ,NE), get the Dirichlet values at physical integral points

        # --- get the jump-average, jump-jump vector at the Dirichlet bds --- #
        penalty = 1.0 / (edgeArea[isDirEdge])
        JADir_temp = np.einsum('i, ij, ijpm, jm->jp', ws, uDh, gphi0, nm[isDirEdge], optimize=True)  # (NDirE,ldof)
        JJDir_temp = np.einsum('i, ij, ijp, j, j->jp', ws, uDh, phi0, edgeArea[isDirEdge], penalty,
                               optimize=True)  # (NDirE,ldof)

        # --- construct the final vector --- #
        NC = mesh.number_of_cells()
        ldof = smspace.number_of_local_dofs()
        shape = (NC, ldof)  # shape.shape: (NC,ldof)
        JADir = np.zeros(shape, dtype=np.float)
        JJDir = np.zeros(shape, dtype=np.float)

        np.add.at(JADir, edge2cell[isDirEdge, 0], JADir_temp)
        np.add.at(JJDir, edge2cell[isDirEdge, 0], JJDir_temp)

        gdof = smspace.number_of_global_dofs()

        return JADir.reshape(gdof, ), JJDir.reshape(gdof, )
