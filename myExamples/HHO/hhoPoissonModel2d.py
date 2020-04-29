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

        NE = self.mesh
        NCE = self.mesh.number_of_edges_of_cells()  # (NC,)

        smldof = self.smspace.number_of_local_dofs()
        psmldof = self.smspace.number_of_local_dofs(p=p + 1)
        eldof = p + 1

        ME = np.zeros((NE*eldof, NE*eldof), dtype=np.float)
        VE = np.zeros((NE*eldof, 1), dtype=np.float)


        def s_c(x):
            # # x[0], the left matrix in current cell
            # # x[1], the right vector in current cell
            # # x[2], the edges index in current cell
            LM_C = x[0]  # the left matrix at this cell
            RV_C = x[1]
            CurrE = x[2]

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
            m = CinvA@B - D
            v = CinvA@RV_C.reshape(-1, 1)

            # --- get the dof location --- #
            


            # --- add to the global matrix and vector --- #
            np.add.at(ME, edge2cell[:, 0], M)













