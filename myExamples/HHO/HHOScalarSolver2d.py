#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: HHOScalarSolver2d.py
# @Author: Yongchao Zhang
# @E-mail: yoczhang@126.com
# @Time: May 04, 2020
# ---


import numpy as np
from HHOBoundaryCondition import HHOBoundaryCondition
from numpy.linalg import inv
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


class HHOScalarSolver2d:
    def __init__(self, pdemodel, uh):
        self.pdemodel = pdemodel
        self.space = pdemodel.space
        self.smspace = pdemodel.smspace
        self.dof = pdemodel.dof
        self.p = pdemodel.p
        self.pde = pdemodel.pde
        self.mesh = pdemodel.mesh
        self.uh = uh

        self.lM = pdemodel.get_left_matrix()  # list, its len is NC, each-term.shape (Cldof,Cldof)
        self.RV = pdemodel.get_right_vector()  # (NC,ldof)

        self.NC = pdemodel.mesh.number_of_cells()
        self.NE = pdemodel.mesh.number_of_edges()
        self.NCE = pdemodel.mesh.ds.number_of_edges_of_cells()  # (NC,)

    def solving_by_static_condensation(self):
        """
        Solving system by static condensation
        """
        p = self.p

        lM = self.lM  # list, its len is NC, each-term.shape (Cldof,Cldof)
        RV = self.RV  # (NC,ldof)

        NC = self.NC
        NE = self.NE
        # NCE = self.NCE  # (NC,)
        # if isinstance(NCE, int):
        #     NCE = NCE*np.ones((NC,), dtype=int)
        cell2edge = self.mesh.ds.cell_to_edge()

        smldof = self.smspace.number_of_local_dofs()
        eldof = p + 1
        egdof = NE * eldof

        # ME = np.zeros((egdof, egdof), dtype=np.float)
        # VE = np.zeros((egdof, 1), dtype=np.float)

        def s_c_solver(x):
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
            edofs = eldof * idx_E.reshape(-1, 1) + np.arange(eldof)
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

        MV = sum(list(map(s_c_solver, zip(lM, RV, cell2edge)))).todense()

        M = MV[:egdof, :egdof]
        V = MV[:, -1]

        # --- treat Dirichlet boundary condition --- #
        hhobc = HHOBoundaryCondition(self.space, self.pde.dirichlet)
        MD, VD = hhobc.applyDirichletBC(M, V)

        # --- solve the edges system --- #
        ub = np.linalg.solve(MD, VD)  # (egdof,)

        # --- solve the cell dofs --- #
        def cell_solver(x):
            # # x[0], the left matrix in current cell
            # # x[1], the right vector in current cell
            # # x[2], the edges index in current cell
            LM_C = x[0]  # the left matrix at this cell
            RV_C = x[1]
            idx_E = x[2]

            A = LM_C[:smldof, :smldof]
            B = LM_C[:smldof, smldof:]

            # --- get the dof location --- #
            edofs = eldof * idx_E.reshape(-1, 1) + np.arange(eldof)
            edofs = np.squeeze(np.transpose(edofs.reshape(1, -1)))  # (NCEdof,)

            # --- interpretation --- #
            # # in the following A, B, C, D matrix,
            # # [ A B ] [u_T] = [f_T]
            # # [ C D ] [u_F] = [ 0 ]
            # # u_T = invA*(f_T - B*u_F)
            invA = inv(A)
            ubcell = ub[edofs]

            u0cell = invA@(RV_C.reshape(-1, 1) - B@ubcell)  # (smldof,1)
            return np.array(np.concatenate([u0cell, ubcell]))

        self.uh[:] = (np.concatenate(list(map(cell_solver, zip(lM, RV, cell2edge))))).flatten()

    def solving_by_direct(self):
        dof = self.dof
        gdof = dof.number_of_global_dofs()
        NC = self.space.mesh.number_of_cells()
        ldof = self.smspace.number_of_local_dofs()

        lM = self.lM  # list, its len is NC, each-term.shape (Cldof,Cldof)
        RV = self.RV  # (NC,ldof)

        cell2dof, doflocation = dof.cell_to_dof()
        cell2dof_split = np.hsplit(cell2dof, doflocation[1:-1])

        def global_matrix(x):
            # # x[0], the left matrix in current cell
            # # x[1], the right vector in current cell
            # # x[2], the dofs-index in current cell
            LM_C = x[0]  # the left matrix at this cell
            RV_C = x[1]
            dof_C = x[2]  # (NCdof,)
            Ndof_C = len(dof_C)

            lenRV_V = len(RV_C)
            RV_C = np.concatenate([RV_C, np.zeros(Ndof_C-lenRV_V,)])

            # --- get the row and col index --- #
            rowIndex = np.einsum('i, k->ik', dof_C, np.ones(Ndof_C,))
            colIndex = np.transpose(rowIndex)
            addcol = gdof*np.ones((colIndex.shape[0], 1))

            rowIndex = np.concatenate([rowIndex, dof_C.reshape(-1, 1)], axis=1)
            colIndex = np.concatenate([colIndex, addcol], axis=1)

            # --- add to the global matrix and vector --- #
            m_v = np.concatenate([LM_C, RV_C.reshape(-1, 1)], axis=1)
            r = csr_matrix((m_v.flat, (rowIndex.flat, colIndex.flat)), shape=(gdof, gdof + 1))
            return r

        MV = sum(list(map(global_matrix, zip(lM, RV, cell2dof_split))))

        M = MV[:gdof, :gdof]
        V = MV[:, -1]

        hhobc = HHOBoundaryCondition(self.space, self.pde.dirichlet)
        MD, VD = hhobc.applyDirichletBC(M, V.todense(), Ncelldof=NC*ldof)

        R = spsolve(MD, VD)  # note that, in R, the dofs are arrangeed as [celldofs, edgedofs]

        self.uh[:] = R[cell2dof]  # rearrange the values by the cell2dof



