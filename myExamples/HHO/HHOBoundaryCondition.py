#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: HHOBoundaryCondition.py
# @Author: Yongchao Zhang
# @E-mail: yoczhang@126.com
# @Time: May 01, 2020
# ---

import numpy as np
from numpy.linalg import inv
from scipy.sparse import spdiags
from fealpy.quadrature import GaussLegendreQuadrature


class HHOBoundaryCondition:
    def __init__(self, space, uD):
        self.space = space
        self.uD = uD
        self.mesh = space.mesh
        self.p = space.p
        self.eldof = self.p + 1
        self.NE = self.mesh.number_of_edges()
        self.egdof = self.NE * self.eldof

    def set_Dirichlet_edge(self):
        mesh = self.mesh
        edge2cell = mesh.ds.edge_to_cell()
        isBdEdge = (edge2cell[:, 0] == edge2cell[:, 1])  # (NE,), the bool vars, to get the boundary edges

        isDirEdge = isBdEdge  # here, we set all the boundary edges are Dir edges

        return isDirEdge

    def set_Neumann_edge(self):
        mesh = self.mesh
        edge2cell = mesh.ds.edge_to_cell()
        bdEdge = (edge2cell[:, 0] == edge2cell[:, 1])  # the bool vars, to get the boundary edges

        isNeuEdge = bdEdge  # here, we first set all the boundary edges are Neu edges

        issetNeuEdge = 'no'
        if issetNeuEdge == 'no':
            isNeuEdge = None

        return isNeuEdge

    def set_Dirichlet_dof(self):
        NE = self.NE
        eldof = self.eldof
        egdof = self.egdof

        isDirDof = np.zeros(egdof).astype(np.bool)  # 1-D array, (egdof,)

        isDirEdge = self.set_Dirichlet_edge()  # (NE,)
        idxDirEdge, = np.nonzero(isDirEdge)  # (NE_Dir,)
        DirDof = eldof * idxDirEdge.reshape(-1, 1) + np.arange(eldof)
        DirDof = np.squeeze(DirDof.reshape(1, -1))  # np.squeeze transform 2-D array (NDirDof,1) into 1-D (NDirDof,)
        isDirDof[DirDof] = True  # 1-D array, (egdof,)

        return isDirDof

    def set_Neumann_dof(self):
        NE = self.NE
        eldof = self.eldof
        egdof = self.egdof

        isNeuDof = np.zeros(egdof).astype(np.bool)  # 1-D array, (egdof,)

        isNeuEdge = self.set_Neumann_edge()

        if isNeuEdge is not None:
            idxNeuEdge, = np.nonzero(isNeuEdge)
            NeuDof = eldof * idxNeuEdge.reshape(-1, 1) + np.arange(eldof)
            NeuDof = np.squeeze(NeuDof.reshape(1, -1))  # np.squeeze transform 2-D array (NNeuDof,1) into 1-D (NNeuDof,)
            isNeuDof[NeuDof] = True  # 1-D array, (egdof,)

        return isNeuDof

    def set_Free_dof(self):
        isDirDof = self.set_Dirichlet_dof()  # (egdof,)
        isNeuDof = self.set_Neumann_dof()  # (egdof,)

        isFreeDof = ~(isDirDof + isNeuDof)  # (egdof,)

        return isFreeDof

    def applyDirichletBC(self, A, b, Ncelldof=0):
        uD = self.uD
        mesh = self.mesh
        smspace = self.space.smspace
        p = self.p
        NE = self.NE
        egdof = self.egdof

        if len(b) != egdof:
            Ncelldof = len(b) - egdof
        Ndof = Ncelldof + egdof  # aim to get the number of all dofs in both cells and edges
        assert Ndof == len(b), 'Ndof should equal to len(b)'

        node = mesh.entity('node')
        edge = mesh.entity('edge')
        hE = mesh.edge_length()

        qf = GaussLegendreQuadrature(p + 1)  # the integral points on edges (1D)
        bcs, ws = qf.quadpts, qf.weights  # bcs.shape: (NQ,2); ws.shape: (NQ,)
        ps = np.einsum('ij, kjm->ikm', bcs, node[edge])  # ps.shape: (NQ,NE,2), NE is the number of edges

        ephi = smspace.edge_basis(ps, index=None, p=p)  # (NQ,NE,eldof), eldof is the number of local 1D dofs on one edge
        EM = smspace.edge_mass_matrix()  # (NE,eldof,eldof), eldof is the number of local 1D dofs on one edge
        invEM = inv(EM)  # (NE,eldof,eldof)

        # ---               set isDirDof                --- #
        # --- and modify the isDirDof based on the Ndof --- #
        isDirEdge = self.set_Dirichlet_edge()  # (NE,)
        isDirDof = self.set_Dirichlet_dof()  # (egdof,)
        isCellDof = np.zeros((Ncelldof,)).astype(np.bool)
        isDirEdge = np.concatenate([isCellDof, isDirEdge])

        # --- project uD to uDP on Dirichlet edges --- #
        uDI = uD(ps[:, isDirEdge, :])  # (NQ,NE_Dir), get the Dirichlet values at physical integral points
        uDrhs = np.einsum('i, ij, ijm, j->jm', ws, uDI, ephi[:, isDirEdge, :], hE[isDirEdge])  # (NE_Dir,eldof)
        uDP = np.einsum('ijk, ik->ij', invEM[isDirEdge], uDrhs)  # (NE_Dir,eldof,eldof)x(NE_Dir,eldof)=>(NE_Dir,eldof)
        uDP = np.squeeze(uDP.reshape(1, -1))  # (NE_Dir,eldof)=>(NE_Dir*eldof,)

        # --- apply to the left-matrix and right-vector --- #
        x = np.zeros((Ndof,), dtype=np.float)
        x[isDirDof] = uDP
        b -= A@x
        bdIdx = np.zeros(Ndof, dtype=np.int)
        bdIdx[isDirDof] = 1
        Tbd = spdiags(bdIdx, 0, Ndof, Ndof)
        T = spdiags(1 - bdIdx, 0, Ndof, Ndof)
        A = T@A@T + Tbd

        b[isDirDof] = x[isDirDof]
        return A, b


