#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: HHOSolver.py
# @Author: Yongchao Zhang, Northwest University, Xi'an
# @E-mail: yoczhang@nwu.edu.cn
# @Site:
# @Time: Sep 29, 2020
# ---

__doc__ = """
The solver hybrid high-order (HHO) method.
"""

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye, bmat


class HHOSolver:
    def __init__(self, M, R, space, pde):
        self.M = M
        self.R = R
        self.space = space
        self.pde = pde
        self.p = self.space.p
        self.mesh = self.space.mesh
        self.NC = self.mesh.number_of_cells()
        self.NE = self.mesh.number_of_edges()

    def StokesSolver(self):
        vSpace = self.space.vSpace
        nu = self.pde.nu
        M = self.M
        R = self.R
        p = self.p
        mesh = self.mesh
        NC = self.NC
        NE = self.NE

        uTlNdof = (p + 1) * (p + 2) // 2
        uFlNdof = p + 1
        pTlNdof = (p + 1) * (p + 2) // 2

        uTgNdof = NC * uTlNdof
        uFgNdof = NE * uFlNdof
        ugNdof = uTgNdof + uFgNdof
        pTgNdof = NC * pTlNdof

        MA1_00 = M[:uTgNdof, :uTgNdof]
        MA1_0b = M[:uTgNdof, uTgNdof:ugNdof]
        MA1_b0 = M[uTgNdof:ugNdof, :uTgNdof]
        MA1_bb = M[uTgNdof:ugNdof, uTgNdof:ugNdof]

        MA2_00 = MA1_00.copy()
        MA2_0b = MA1_0b.copy()
        MA2_b0 = MA1_b0.copy()
        MA2_bb = MA1_bb.copy()

        MB1_00 = M[2*ugNdof:(2*ugNdof + pTgNdof), :uTgNdof]
        MB1_0b = M[2*ugNdof:(2*ugNdof + pTgNdof), uTgNdof:ugNdof]
        MB2_00 = M[2*ugNdof:(2*ugNdof + pTgNdof), ugNdof:(ugNdof + uTgNdof)]
        MB2_0b = M[2*ugNdof:(2*ugNdof + pTgNdof), (ugNdof + uTgNdof):2*ugNdof]

        MB1_00t = M[:uTgNdof, 2*ugNdof:(2*ugNdof + pTgNdof)]
        MB1_b0 = M[uTgNdof:ugNdof, 2*ugNdof:(2*ugNdof + pTgNdof)]
        MB2_00t = M[ugNdof:(ugNdof + uTgNdof), 2*ugNdof:(2*ugNdof + pTgNdof)]
        MB2_b0 = M[(ugNdof + uTgNdof):2*ugNdof, 2*ugNdof:(2*ugNdof + pTgNdof)]

        # # the Lagrange multiplier vector (for the pressure condition: \int p = 0)
        L = M[-1, 2 * ugNdof:2 * ugNdof + pTgNdof]
        Lt = M[2*ugNdof:2*ugNdof + pTgNdof, -1]

        pphi = vSpace.basis  # (NQ,NC,pldof)
        intp = vSpace.integralalg.integral(pphi, celltype=True)  # (NC,pldof)

        LF = intp[:, 0][np.newaxis, :]  # (1,NC)
        LT = intp[:, 1:].reshape(1, -1)  # (1,...)
        L = bmat([[csr_matrix((1, 2*uTgNdof)), LT, csr_matrix((1, 2*uFgNdof)), LF]], format='csr')

        # --- to reconstruct the global matrix
        pgIdx = np.arange(0, pTgNdof, pTlNdof)  # (NC,)
        MB1_bb = MB1_0b[pgIdx, :]  # (NC,uFgNdof)
        MB2_bb = MB2_0b[pgIdx, :]  # (NC,uFgNdof)
        MB1_bbt = MB1_b0[:, pgIdx]  # (uFgNdof,NC)
        MB2_bbt = MB2_b0[:, pgIdx]  # (uFgNdof,NC)

        MB1_00 = np.delete(MB1_00.todense(), pgIdx, axis=0)
        MB1_0b = np.delete(MB1_0b.todense(), pgIdx, axis=0)
        MB2_00 = np.delete(MB2_00.todense(), pgIdx, axis=0)
        MB2_0b = np.delete(MB2_0b.todense(), pgIdx, axis=0)

        MB1_00t = np.delete(MB1_00t.todense(), pgIdx, axis=1)
        MB1_b0 = np.delete(MB1_b0.todense(), pgIdx, axis=1)
        MB2_00t = np.delete(MB2_00t.todense(), pgIdx, axis=1)
        MB2_b0 = np.delete(MB2_b0.todense(), pgIdx, axis=1)

        # # get global matrix
        VP_F0 = np.zeros((NC, uTgNdof))  # here, NC is also the number of global dofs of pressure
        LF0 = np.zeros((1, uFgNdof))
        LT0 = np.zeros((1, uTgNdof))
        A = bmat([[MA1_00, None, MB1_00t], [None, MA2_00, MB2_00t], [MB1_00, MB2_00, None]], format='csr')
        B = bmat([[MA1_0b, None, VP_F0.T, LT0.T], [None, MA2_0b, VP_F0.T, LT0.T], [MB1_0b, MB2_0b, None, LT.T]], format='csr')
        C = bmat([[MA1_b0, None, MB1_b0], [None, MA2_b0, MB2_b0], [VP_F0, VP_F0, None], [LT0, LT0, LT]], format='csr')
        D = bmat([[MA1_bb, None, MB1_bbt, LF0.T], [None, MA2_bb, MB2_bbt, LF0.T], [MB1_bb, MB2_bb, None, LF.T], [LF0, LF0, LF, None]], format='csr')

        # # to get the inv A
        cell2dof, doflocation = vSpace.dof.cell_to_dof()
        StiffM = vSpace.reconstruction_stiff_matrix()  # list, (NC,), each-term.shape (Cldof,Cldof)
        StabM = vSpace.reconstruction_stabilizer_matrix()  # list, (NC,), each-term.shape (Cldof,Cldof)
        divM0, divM1 = self.space.cell_divergence_matrix()  # e.g., divM0, list, (NC,); divM0: (pTlNdof,\sum_C{Cldof})
        divM0_split = np.hsplit(divM0, doflocation[1:-1])
        divM1_split = np.hsplit(divM1, doflocation[1:-1])

        AlNdof = 2*uTlNdof + pTlNdof - 1
        Al = np.zeros((AlNdof, AlNdof), dtype=np.float)
        invA = np.zeros((NC*AlNdof, NC*AlNdof), dtype=np.float)

        def func_invA(x):
            stiffM_TT = nu * x[0][:uTlNdof, :uTlNdof]
            stabM_TT = nu * x[1][:uTlNdof, :uTlNdof]
            divM0_TT = x[2][1:, :uTlNdof]
            divM1_TT = x[3][1:, :uTlNdof]
            CIdx = x[4]

            Al[:uTlNdof, :uTlNdof] = stiffM_TT + stabM_TT
            Al[uTlNdof:2*uTlNdof, uTlNdof:2*uTlNdof] = stiffM_TT + stabM_TT
            Al[2*uTlNdof:, :uTlNdof] = divM0_TT
            Al[2*uTlNdof:, uTlNdof:2*uTlNdof] = divM1_TT
            Al[:uTlNdof, 2*uTlNdof:] = divM0_TT.T
            Al[uTlNdof:2*uTlNdof, 2*uTlNdof:] = divM1_TT.T

            # u0Tdof_C = CIdx * uTlNdof + np.arange(uTlNdof)
            # u0T_rowIdx = np.einsum('i, k->ik', u0Tdof_C, np.ones(AlNdof, ))
            # u1T_rowIdx = uTgNdof + u0T_rowIdx
            # pT_rowIdx = 2*uTgNdof + u0T_rowIdx
            # rowIdx = np.concatenate([u0T_rowIdx, u1T_rowIdx, pT_rowIdx], axis=0)
            # colIdx = rowIdx.T

            u0Idx = CIdx * uTlNdof + np.arange(uTlNdof)
            u1Idx = uTgNdof + u0Idx
            pIdx = 2*uTgNdof + CIdx * (pTlNdof - 1) + np.arange(pTlNdof - 1)
            Idx = np.concatenate([u0Idx, u1Idx, pIdx])

            i, j = np.ix_(Idx, Idx)
            invA[i, j] = np.linalg.inv(Al)
            return None

        t = list(map(func_invA, zip(StiffM, StabM, divM0_split, divM1_split, range(NC))))  # TODO: why here list() is necessary?
        # tt = np.max(abs(np.linalg.inv(A.todense())-invA))
        # print("max( abs(invA - inv(A)) ) = ", tt)

        print("solve system:")

    def StokesSolver_1(self):
        vSpace = self.space

        # --- the Poisson (scalar) HHO space
        gdof_scal = self.space.dof.number_of_global_dofs()
        StiffM = self.space.reconstruction_stiff_matrix()  # list, its len is NC, each-term.shape (Cldof,Cldof)
        StabM = self.space.reconstruction_stabilizer_matrix()  # list, its len is NC, each-term.shape (Cldof,Cldof)

        # --- the divergence term
        divM0, divM1 = self.space.cell_divergence_matrix()  # divM0, list, (NC,); divM1, list, (NC,)






















