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

        uTgNdof = NC * (p + 1) * (p + 2) // 2
        uFgNdof = NE * (p + 1)
        ugNdof = uTgNdof + uFgNdof
        pTgNdof = NC * (p + 1) * (p + 2) // 2

        MA1_00 = M[:uTgNdof, :uTgNdof]
        MA1_0b = M[:uTgNdof, uTgNdof:ugNdof]
        MA1_b0 = M[uTgNdof:ugNdof, :uTgNdof]
        MA1_bb = M[uTgNdof:ugNdof, uTgNdof:ugNdof]

        MA2_00 = np.copy(MA1_00)
        MA2_0b = np.copy(MA1_0b)
        MA2_b0 = np.copy(MA1_b0)
        MA2_bb = np.copy(MA1_bb)

        MB1_0 = M[:uTgNdof, 2*ugNdof:2*ugNdof + pTgNdof]
        MB1_b = M[uTgNdof:ugNdof, 2*ugNdof:2*ugNdof + pTgNdof]
        











