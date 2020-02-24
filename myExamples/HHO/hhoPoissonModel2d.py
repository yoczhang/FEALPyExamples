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

        return lM

    def get_right_vector(self):
        space = self.space
        f = self.pde.source
        lV = space.source_vector(f)  # (NC,ldof)

        return lV  # (NC,ldof)

    def solving_by_static_condensation(self):
        """
        Solving system by static condensation
        """
        p = self.p

        lM = self.get_left_matrix()
        lV = self.get_right_vector()

        NCE = self.mesh.number_of_edges_of_cells()  # (NC,)

        smldof = self.smspace.number_of_local_dofs()
        eldof = p + 1

        def s_c(x):
            # # x[0], the left matrix
            # # x[1], the right vector
            # # x[2], the number of edges of current cell
            A = x[0][:smldof, :smldof]
            B = x[0][:smldof, smldof:]
            C = x[0][smldof:, :smldof]
            D = x[0][smldof:, smldof:]








