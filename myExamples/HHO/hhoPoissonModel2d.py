#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: hhoPoissonModel2d.py
# @Author: Yongchao Zhang
# @E-mail: yoczhang@126.com
# @Time: Feb 24, 2020
# ---

import numpy as np

from HHOScalarSpace2d import HHOScalarSpace2d
from numpy.linalg import inv
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from timeit import default_timer as timer


class hhoPoissonModel2d(object):
    def __init__(self, pde, mesh, p):
        self.p = p
        self.space = HHOScalarSpace2d(mesh, p)
        self.smspace = self.space.smspace
        self.mesh = self.space.mesh
        self.dof = self.space.dof
        self.pde = pde
        self.uh = self.space.function()
        self.integralalg = self.space.integralalg

    def get_left_matrix(self):
        space = self.space
        StiffM = space.reconstruction_stiff_matrix()  # list, its len is NC, each-term.shape (Cldof,Cldof)
        StabM = space.reconstruction_stabilizer_matrix()  # list, its len is NC, each-term.shape (Cldof,Cldof)

        f = lambda x: x[0]+x[1]
        lM = list(map(f, zip(StiffM, StabM)))

        return lM  # list, its len is NC, each-term.shape (Cldof,Cldof)

    def get_right_vector(self):
        space = self.space
        f = self.pde.source
        RV = space.source_vector(f)  # (NC,ldof)

        return RV  # (NC,ldof)

    def solving_by_static_condensation(self):
        """
        Solving system by static condensation
        """
        p = self.p

        lM = self.get_left_matrix()  # list, its len is NC, each-term.shape (Cldof,Cldof)
        RV = self.get_right_vector()  # (NC,ldof)

        NE = self.mesh.number_of_edges()
        NCE = self.mesh.number_of_edges_of_cells()  # (NC,)

        smldof = self.smspace.number_of_local_dofs()
        psmldof = self.smspace.number_of_local_dofs(p=p + 1)
        eldof = p + 1
        egdof = NE * eldof

        # ME = np.zeros((egdof, egdof), dtype=np.float)
        # VE = np.zeros((egdof, 1), dtype=np.float)

        def s_c(x):
            # # x[0], the left matrix in current cell
            # # x[1], the right vector in current cell
            # # x[2], the edges index in current cell
            LM_C = x[0]  # the left matrix at this cell
            RV_C = x[1]
            idx_E = x[2]

            NCEdof = len(idx_E)*eldof

            A = LM_C[:smldof, :smldof]
            B = LM_C[:smldof, smldof:]
            C = LM_C[smldof:, :smldof]
            D = LM_C[smldof:, smldof:]

            # --- interpretation --- #
            # # in the following A, B, C, D matrix,
            # # [ A B ] [u_T] = [f_T]
            # # [ C D ] [u_F] = [ 0 ]
            # # by eliminating the u_T, we have the only u_F variable:
            # # (C*invA*B - D)*u_F = C*invA*f_T
            # # the matrix (C*invA*B - D) and the vector C*invA*f_T are what we want to get in this function
            CinvA = C@inv(A)
            m = CinvA@B - D  # (NCEdof,NCEdof)
            v = CinvA@RV_C.reshape(-1, 1)  # (NCEdof,)
            m_v = np.concatenate([m, v], axis=1)

            # --- get the dof location --- #
            edofs = eldof*idx_E.reshape(-1, 1) + np.arange(eldof)
            edofs = np.transpose(edofs.reshape(1, -1))  # (NCEdof,1)

            rowIndex = np.einsum('ij, k->ik', edofs, np.ones(NCEdof))
            colIndex = np.transpose(rowIndex)
            addcol = egdof*np.ones((colIndex.shape[0], 1))

            rowIndex = np.concatenate([rowIndex, edofs], axis=1)
            colIndex = np.concatenate([colIndex, addcol], axis=1)

            # --- add to the global matrix and vector --- #
            # global ME, VE
            # ME[rowIndex.flat, colIndex.flat] += m.flat
            # VE[rowIndex[:, 0]] += v
            r = csr_matrix((m_v.flat, (rowIndex.flat, colIndex.flat)), shape=(egdof, egdof+1))
            return r

        MV = sum(list(map(s_c, zip(lM, RV, NCE)))).todense()

        M = MV[:egdof, :egdof]
        V = MV[:, -1]

        # --- treat Dirichlet boundary condition --- #
        isDirEdge = self.set_Dirichlet_edge()
        idxDirEdge = np.arange(NE)[isDirEdge]
        dofDir = eldof*idxDirEdge.reshape(-1, 1) + np.arange(eldof)
        dofDir = np.squeeze(dofDir.reshape(1, -1))  # np.squeeze transform 2-D array (NdofDir,1) into 1-D (NdofDir,)
        dofFree = np.ones(egdof).astype(np.bool)  # 1-D array, (egdof,)
        dofFree[dofDir] = False  # 1-D array, (egdof,)

        V = V - M[:, dofDir]@V[dofDir]
        # --- solve the edges system --- #
        # np.linalg.solve(A, b)

    def set_Dirichlet_edge(self):
        mesh = self.mesh
        edge2cell = mesh.ds.edge_to_cell()
        bdEdge = (edge2cell[:, 0] == edge2cell[:, 1])  # the bool vars, to get the boundary edges

        isDirEdge = bdEdge  # here, we set all the boundary edges are Dir edges

        return isDirEdge















