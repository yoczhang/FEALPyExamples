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
    def __init__(self, M, R, space):
        self.M = M
        self.R = R
        self.space = space
        self.p = self.space.p
        self.mesh = self.space.mesh
        self.NC = self.mesh.number_of_cells()
        self.NE = self.mesh.number_of_edges()

    def StokesSolver(self):
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

        L = M[-1, 2 * ugNdof:2 * ugNdof + pTgNdof]  # the Lagrange multiplier vector (for the pressure condition: \int p = 0).
        Lt = M[2*ugNdof:2*ugNdof + pTgNdof, -1]

        # --- to reconstruct the global matrix
        pgIdx = np.arange(0, pTgNdof, pTlNdof)  # (NC,)
        MB1_bb = MB1_0b[pgIdx, :]  # (NC,uFgNdof)
        MB2_bb = MB2_0b[pgIdx, :]  # (NC,uFgNdof)
        MB1_bbt = MB1_bb[:, pgIdx]  # (uFgNdof,NC)
        MB2_bbt = MB2_bb[:, pgIdx]  # (uFgNdof,NC)

        MB1_00 = np.delete(MB1_00, pgIdx, axis=0)
        MB1_0b = np.delete(MB1_0b, pgIdx, axis=0)
        MB2_00 = np.delete(MB2_00, pgIdx, axis=0)
        MB2_0b = np.delete(MB2_0b, pgIdx, axis=0)

        MB1_00t = np.delete(MB1_00t, pgIdx, axis=1)
        MB1_b0 = np.delete(MB1_b0, pgIdx, axis=1)
        MB2_00t = np.delete(MB2_00t, pgIdx, axis=1)
        MB2_b0 = np.delete(MB2_b0, pgIdx, axis=1)



        uT0 = csr_matrix((uTgNdof, uTgNdof))
        LT0 = csr_matrix((uTgNdof, 1))
        pT0 = csr_matrix((pTgNdof, pTgNdof))
        A = bmat([[MA1_00, uT0, MB1_00t, LT0], [uT0, MA2_00, MB2_00t, LT0], [MB1_00, MB2_00, pT0, L], [LT0.T, LT0.T, Lt, 0]], format='csr')
        A0 = bmat([[MA1_00, uT0], [uT0, MA2_00]], format='csr')

        # --- test
        bb = bmat([[MB1_0], [MB2_0], [pT0], [Lt]], format='csr')

        # np.linalg.matrix_rank

        print("solve system:")

    def StokesSolver_1(self):
        space = self.space

        # --- the Poisson (scalar) HHO space
        gdof_scal = space.vSpace.dof.number_of_global_dofs()
        StiffM = space.vSpace.reconstruction_stiff_matrix()  # list, its len is NC, each-term.shape (Cldof,Cldof)
        StabM = space.vSpace.reconstruction_stabilizer_matrix()  # list, its len is NC, each-term.shape (Cldof,Cldof)

        # --- the divergence term
        divM0, divM1 = space.cell_divergence_matrix()  # divM0, list, (NC,); divM1, list, (NC,)






















