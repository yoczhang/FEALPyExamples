#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: StokesHHOModel2d.py
# @Author: Yongchao Zhang
# @E-mail: yoczhang@126.com
# @Time: May 20, 2020
# ---


import numpy as np
from HHOStokesSpace2d import HHOStokesSapce2d
from fealpy.quadrature import GaussLegendreQuadrature
from scipy.sparse import spdiags
from HHOBoundaryCondition_new import HHOBoundaryCondition
from numpy.linalg import inv
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from timeit import default_timer as timer


class StokesHHOModel2d:
    def __init__(self, pde, mesh, p):
        self.p = p
        self.mesh = mesh
        self.pde = pde
        self.space = HHOStokesSapce2d(mesh, p)
        self.integralalg = self.space.integralalg

    def get_left_matrix(self):
        M = self.space.system_matrix(self.pde.nu)
        return M

    def get_right_vector(self):
        V = self.space.system_source(self.pde.source)
        return V

    def solve(self, solver='direct'):
        A = self.space.system_matrix(self.pde.nu)  # (2*vgdof+pgdof+1,2*vgdof+pgdof+1)
        b = self.space.system_source(self.pde.source)  # (2*vgdof+pgdof+1,1)

        AD, bD = self.applyDirichletBC(A, b)



    def applyDirichletBC(self, A, b):
        p = self.p
        mesh = self.mesh
        uD = self.pde.dirichlet  # uD(bcs): (NQ,NC,ldof,2)
        vgdof = self.space.vSpace.number_of_global_dofs()
        pgdof = self.space.pSpace.number_of_global_dofs()
        idxDirEdge = self.setDirichletEdges()
        idxDirDof = self.setStokesDirichletDofs()
        node = mesh.entity('node')
        edge = mesh.entity('edge')
        hE = mesh.edge_length()

        # --- --- #
        Nalldof = 2*vgdof+pgdof+1
        assert Nalldof == len(b), 'Nalldof should equal to len(b)'

        # --- --- #
        qf = GaussLegendreQuadrature(p + 3)  # the integral points on edges (1D)
        bcs, ws = qf.quadpts, qf.weights  # bcs.shape: (NQ,2); ws.shape: (NQ,)
        ps = np.einsum('ij, kjm->ikm', bcs, node[edge])  # ps.shape: (NQ,NE,2), NE is the number of edges

        vephi = self.space.vSpace.edge_basis(ps, index=None, p=p)
        # # (NQ,NE,eldof), eldof is the number of local 1D dofs on one edge
        invEM = self.space.vSpace.invEM  # (NE,eldof,eldof)

        # --- project uD to uDP on Dirichlet edges --- #
        uDI = uD(ps[:, idxDirEdge, :])  # (NQ,NE_Dir,2), get the Dirichlet values at physical integral points
        uDrhs = np.einsum('i, ijn, ijm, j->jmn', ws, uDI, vephi[:, idxDirEdge, :], hE[idxDirEdge])  # (NE_Dir,eldof,2)
        uDP = np.einsum('ijk, ikn->ijn', invEM[idxDirEdge, ...], uDrhs)  # (NE_Dir,eldof,eldof)x(NE_Dir,eldof,2)=>(NE_Dir,eldof,2)
        uDP0 = uDP[..., 0]
        uDP1 = uDP[..., 1]

        # --- apply to the left-matrix and right-vector --- #
        x = np.zeros((Nalldof, 1), dtype=np.float)
        x[idxDirDof, 0] = np.concatenate([uDP0.flatten(), uDP1.flatten()])
        b -= A @ x
        bdIdx = np.zeros(Nalldof, dtype=np.int)
        bdIdx[idxDirDof] = 1
        Tbd = spdiags(bdIdx, 0, Nalldof, Nalldof)
        T = spdiags(1 - bdIdx, 0, Nalldof, Nalldof)
        A = T @ A @ T + Tbd

        b[idxDirDof] = x[idxDirDof]
        return A, b

    def setStokesDirichletDofs(self):
        eldof = self.p + 1
        NC = self.mesh.number_of_cells()
        vldof = self.space.vSpace.smspace.number_of_local_dofs()
        vgdof = self.space.vSpace.number_of_global_dofs()
        pgdof = self.space.pSpace.number_of_global_dofs()
        idxDirEdge = self.setDirichletEdges()
        idxDirDof0 = eldof * idxDirEdge.reshape(-1, 1) + np.arange(eldof)
        idxDirDof0 = np.squeeze(idxDirDof0.reshape(1, -1))  # np.squeeze transform 2-D array (NDirDof,1) into 1-D (NDirDof,)

        Ncelldofs = NC*vldof
        idxDirDof0 += Ncelldofs
        idxDirDof1 = vgdof + idxDirDof0

        return np.concatenate([idxDirDof0, idxDirDof1])

    def setDirichletEdges(self):
        mesh = self.mesh
        edge2cell = mesh.ds.edge_to_cell()
        isBdEdge = (edge2cell[:, 0] == edge2cell[:, 1])  # (NE,), the bool vars, to get the boundary edges

        isDirEdge = isBdEdge  # here, we set all the boundary edges are Dir edges
        idxDirEdge, = np.nonzero(isDirEdge)  # (NE_Dir,)
        return idxDirEdge










