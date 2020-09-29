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
    def __init__(self, A, R, space):
        self.A = A
        self.R = R
        self.space = space
        self.p = self.space.p
        self.mesh = self.space.mesh
        self.NC = self.mesh.number_of_cells()
        self.NE = self.mesh.number_of_edges()

    def StokesSolver(self):
        A = self.A
        R = self.R
        p = self.p
        mesh = self.mesh
        NC = self.NC
        NE = self.NE

        gNdof_uT = NC * (p + 1) * (p + 2) // 2
        gNdof_uF = NE * (p + 1)
        gNdof_pT = NC * (p + 1) * (p + 2) // 2

        






