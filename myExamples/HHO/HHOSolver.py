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
from numpy.linalg import solve, inv
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye, bmat
from timeit import default_timer as timer
from multiprocessing.dummy import Pool as ThreadPool


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

        # # 'l' denotes 'local';
        # # 'g' denotes 'global'.
        TlNdof = (p + 1) * (p + 2) // 2
        uTlNdof = TlNdof
        uFlNdof = p + 1
        pTlNdof = TlNdof - 1
        pFlNdof = 1

        uTgNdof = NC * uTlNdof
        uFgNdof = NE * uFlNdof
        ugNdof = uTgNdof + uFgNdof
        pTgNdof = NC * pTlNdof
        pFgNdof = NC * pFlNdof
        pgNdof = pTgNdof + pFgNdof

        # --- matrix setting --- #
        MA1_00 = M[:uTgNdof, :uTgNdof]
        MA1_0b = M[:uTgNdof, uTgNdof:ugNdof]
        MA1_b0 = M[uTgNdof:ugNdof, :uTgNdof]
        MA1_bb = M[uTgNdof:ugNdof, uTgNdof:ugNdof]

        MA2_00 = MA1_00.copy()
        MA2_0b = MA1_0b.copy()
        MA2_b0 = MA1_b0.copy()
        MA2_bb = MA1_bb.copy()

        MB1_00 = M[2*ugNdof:(2*ugNdof + pgNdof), :uTgNdof]
        MB1_0b = M[2*ugNdof:(2*ugNdof + pgNdof), uTgNdof:ugNdof]
        MB2_00 = M[2*ugNdof:(2*ugNdof + pgNdof), ugNdof:(ugNdof + uTgNdof)]
        MB2_0b = M[2*ugNdof:(2*ugNdof + pgNdof), (ugNdof + uTgNdof):2*ugNdof]

        MB1_00t = M[:uTgNdof, 2*ugNdof:(2*ugNdof + pgNdof)]
        MB1_b0 = M[uTgNdof:ugNdof, 2*ugNdof:(2*ugNdof + pgNdof)]
        MB2_00t = M[ugNdof:(ugNdof + uTgNdof), 2*ugNdof:(2*ugNdof + pgNdof)]
        MB2_b0 = M[(ugNdof + uTgNdof):2*ugNdof, 2*ugNdof:(2*ugNdof + pgNdof)]

        # # the right hand side
        r0_0 = R[:uTgNdof, :]
        r0_1 = R[ugNdof:(ugNdof+uTgNdof), :]
        R0 = np.concatenate([r0_0, r0_1, np.zeros((pTgNdof, 1), dtype=np.float)])  # (2*uTgNdof+pTgNdof, 1)
        Rb = np.zeros((2*uFgNdof+pFgNdof+1, 1), dtype=np.float)

        # # the Lagrange multiplier vector (for the pressure condition: \int p = 0)
        # # In HHO-Stokes equation, the velocity and pressure have the same order, here we just use the velocity space.
        pphi = vSpace.basis
        intp = vSpace.integralalg.integral(pphi, celltype=True)  # (NC,pldof)

        LF = intp[:, 0][np.newaxis, :]  # (1,NC)
        LT = intp[:, 1:].reshape(1, -1)  # (1,...)

        # # to reconstruct the global matrix
        pgIdx = np.arange(0, pgNdof, TlNdof)  # (NC,)
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

        # --- get global matrix --- #
        VP_F0 = np.zeros((pFgNdof, uTgNdof))
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
        divM0, divM1 = self.space.cell_divergence_matrix()  # e.g., divM0, list, (NC,); divM0: (TlNdof,\sum_C{Cldof})
        divM0_split = np.hsplit(divM0, doflocation[1:-1])
        divM1_split = np.hsplit(divM1, doflocation[1:-1])

        AlNdof = 2*uTlNdof + pTlNdof
        invA = np.zeros((NC * AlNdof, NC * AlNdof), dtype=np.float)

        def func_invA(x):
            r = None
            stiffM_TT = nu * x[0][:uTlNdof, :uTlNdof]
            stabM_TT = nu * x[1][:uTlNdof, :uTlNdof]
            divM0_TT = x[2][1:, :uTlNdof]
            divM1_TT = x[3][1:, :uTlNdof]
            CIdx = x[4]

            Al = np.zeros((AlNdof, AlNdof), dtype=np.float)

            Al[:uTlNdof, :uTlNdof] = stiffM_TT + stabM_TT
            Al[uTlNdof:2*uTlNdof, uTlNdof:2*uTlNdof] = stiffM_TT + stabM_TT
            Al[2*uTlNdof:, :uTlNdof] = divM0_TT
            Al[2*uTlNdof:, uTlNdof:2*uTlNdof] = divM1_TT
            Al[:uTlNdof, 2*uTlNdof:] = divM0_TT.T
            Al[uTlNdof:2*uTlNdof, 2*uTlNdof:] = divM1_TT.T

            u0Idx = CIdx * uTlNdof + np.arange(uTlNdof)
            u1Idx = uTgNdof + u0Idx
            pIdx = 2 * uTgNdof + CIdx * pTlNdof + np.arange(pTlNdof)
            Idx = np.concatenate([u0Idx, u1Idx, pIdx])

            # # one way assembling, this way is more quickly
            i, j = np.ix_(Idx, Idx)
            invA[i, j] = inv(Al)

            # # other way assembling
            # rowIdx = np.einsum('i, k->ik', Idx, np.ones((AlNdof, ), dtype=np.int))
            # colIdx = rowIdx.T
            # r = csr_matrix((inv(Al).flat, (rowIdx.flat, colIdx.flat)), shape=(NC * AlNdof, NC * AlNdof), dtype=np.float)
            return r

        start = timer()
        # invAt = sum(list(map(func_invA, zip(StiffM, StabM, divM0_split, divM1_split, range(NC)))))
        invAt = list(map(func_invA, zip(StiffM, StabM, divM0_split, divM1_split, range(NC))))
        # pool = ThreadPool()
        # invAt = sum(pool.map(func_invA, zip(StiffM, StabM, divM0_split, divM1_split, range(NC))))
        # pool.close()
        # pool.join()
        end = timer()
        print("  |___ TIME: get inv matrix (cell-by-cell):", end - start)

        # start = timer()
        # invAtt = inv(A)
        # invAtt = np.linalg.inv(A.todense())
        # end = timer()
        # print("get inv matrix (directly):", end - start)
        # print("max( abs(invA - inv(A)) ) = ", np.max(abs(invAtt-invA)))

        # --- solve the Static Condensation system --- #
        # print("solve system:")
        invA = csr_matrix(invA)  # To makesure invA is sparse matrix is important!!!
        stacM = D - C@invA@B
        stacR = Rb - C@invA@R0

        # # deal the Dirichlet boundary edges
        uD = self.pde.dirichlet  # uD(bcs): (NQ,NC,ldof,2)
        idxDirEdge = self.pde.idxDirEdge
        MD, RD = self.space.applyDirichletBC(stacM, stacR, uD, idxDirEdge=idxDirEdge, StaticCondensation=True)

        # # solve the system
        print("  |___ Begin solve 'sparse' algebraic equations:")
        start = timer()
        Xb = spsolve(MD, RD)
        end = timer()
        print("  |___ TIME: in static solver, solve 'sparse' algebraic equations:", end - start)

        X0 = invA@(np.squeeze(R0) - B@Xb)

        # # rearrange the dofs
        X = np.zeros((2*ugNdof + pgNdof + 1,))
        X[:uTgNdof] = X0[:uTgNdof]
        X[uTgNdof:ugNdof] = Xb[:uFgNdof]
        X[ugNdof:ugNdof+uTgNdof] = X0[uTgNdof:2*uTgNdof]
        X[ugNdof+uTgNdof:2*ugNdof] = Xb[uFgNdof:2*uFgNdof]

        if pTlNdof == 0:
            P = Xb[2*uFgNdof:-1].reshape(-1, 1)  # (PFgNdof,1)
        else:
            P0 = X0[2 * uTgNdof:].reshape(-1, pTlNdof)  # (pTgNdof,pTlNdof)
            Pb = Xb[2 * uFgNdof:-1].reshape(-1, 1)  # (PFgNdof,1)
            P = np.concatenate([Pb, P0], axis=1)

        X[2*ugNdof:-1] = P.flatten()
        return X

    def StokesSolver_1(self):
        vSpace = self.space

        # --- the Poisson (scalar) HHO space
        gdof_scal = self.space.dof.number_of_global_dofs()
        StiffM = self.space.reconstruction_stiff_matrix()  # list, its len is NC, each-term.shape (Cldof,Cldof)
        StabM = self.space.reconstruction_stabilizer_matrix()  # list, its len is NC, each-term.shape (Cldof,Cldof)

        # --- the divergence term
        divM0, divM1 = self.space.cell_divergence_matrix()  # divM0, list, (NC,); divM1, list, (NC,)






















