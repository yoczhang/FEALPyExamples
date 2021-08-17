#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: sym_diff_basis.py
# @Author: Yongchao Zhang, Northwest University, Xi'an
# @E-mail: yoczhang@nwu.edu.cn
# @Site:
# @Time: Aug 16, 2021
# ---

from sympy import symbols, cos, sin, diff, lambdify
import numpy as np
import math
from fractions import Fraction
from fealpy.functionspace.femdof import multi_index_matrix2d
from fealpy.mesh import MeshFactory as MF
from fealpy.quadrature import FEMeshIntegralAlg


class compute_basis:
    def __init__(self, p, mesh):
        self.p = p
        self.mesh = mesh
        self.lam0, self.lam1, self.lam2 = symbols('lam0 lam1 lam2')
        self.lam0_x, self.lam1_x, self.lam2_x = symbols('lam0_x lam1_x lam2_x')
        self.lam0_y, self.lam1_y, self.lam2_y = symbols('lam0_y lam1_y lam2_y')

    def basis(self, p, m, n, k):
        L0 = 1
        L1 = 1
        L2 = 1
        for i in range(m):
            L0 *= (p * self.lam0 - i)
        for i in range(n):
            L1 *= (p * self.lam1 - i)
        for i in range(k):
            L2 *= (p * self.lam2 - i)
        C = Fraction(1, math.factorial(m) * math.factorial(n) * math.factorial(k))
        return C * L0 * L1 * L2

    def diff_basis(self):
        basis = self.basis
        p = self.p
        # phi_300 = basis(p, 3, 0, 0)
        # phi_210 = basis(p, 2, 1, 0)
        # phi_201 = basis(p, 2, 0, 1)

        index = multi_index_matrix2d(p)
        n = len(index)
        phi_list = list([])
        phi_x_list = list([])
        phi_xx_list = list([])
        phi_xxx_list = list([])
        phi_yxx_list = list([])
        phi_y_list = list([])
        phi_yy_list = list([])
        phi_xyy_list = list([])
        phi_yyy_list = list([])
        for i in range(n):
            phi_list.append(basis(p, index[i, 0], index[i, 1], index[i, 2]))
            phi_x_list.append(diff(phi_list[i], self.lam0) * self.lam0_x
                              + diff(phi_list[i], self.lam1) * self.lam1_x
                              + diff(phi_list[i], self.lam2) * self.lam2_x)
            phi_xx_list.append(diff(phi_x_list[i], self.lam0) * self.lam0_x
                               + diff(phi_x_list[i], self.lam1) * self.lam1_x
                               + diff(phi_x_list[i], self.lam2) * self.lam2_x)
            phi_xxx_list.append(diff(phi_xx_list[i], self.lam0) * self.lam0_x
                                + diff(phi_xx_list[i], self.lam1) * self.lam1_x
                                + diff(phi_xx_list[i], self.lam2) * self.lam2_x)
            phi_yxx_list.append(diff(phi_xx_list[i], self.lam0) * self.lam0_y
                                + diff(phi_xx_list[i], self.lam1) * self.lam1_y
                                + diff(phi_xx_list[i], self.lam2) * self.lam2_y)

            phi_y_list.append(diff(phi_list[i], self.lam0) * self.lam0_y
                              + diff(phi_list[i], self.lam1) * self.lam1_y
                              + diff(phi_list[i], self.lam2) * self.lam2_y)
            phi_yy_list.append(diff(phi_y_list[i], self.lam0) * self.lam0_y
                               + diff(phi_y_list[i], self.lam1) * self.lam1_y
                               + diff(phi_y_list[i], self.lam2) * self.lam2_y)
            phi_yyy_list.append(diff(phi_yy_list[i], self.lam0) * self.lam0_y
                                + diff(phi_yy_list[i], self.lam1) * self.lam1_y
                                + diff(phi_yy_list[i], self.lam2) * self.lam2_y)
            phi_xyy_list.append(diff(phi_yy_list[i], self.lam0) * self.lam0_x
                                + diff(phi_yy_list[i], self.lam1) * self.lam1_x
                                + diff(phi_yy_list[i], self.lam2) * self.lam2_x)

        # print('end of the func')
        return phi_list, phi_x_list, phi_xx_list, phi_xxx_list, phi_yxx_list, phi_y_list, phi_yy_list, phi_yyy_list, phi_xyy_list

    def get_highorder_diff(self, bcs, Dlambda, p=None):
        """

        :param bcs: (NQ,GD+1), in 2D, (NQ,3)
        :param Dlambda: (NC,GD+1,GD), in 2D, (NC,3,2)
        :return:
        """
        p = self.p if p is None else p

        phi, phi_x, phi_xx, phi_xxx, phi_yxx, phi_y, phi_yy, phi_yyy, phi_xyy = self.diff_basis()
        # basis = self.get_basis_details()
        # Nb = len(basis)
        # ldof = len(basis[0])
        ldof = len(phi)

        NQ = bcs.shape[0]
        NC = Dlambda.shape[0]
        phi_val = np.zeros((NQ, ldof))
        phi_x_val = np.zeros((NQ, NC, ldof))
        phi_xx_val = np.zeros((NQ, NC, ldof))
        phi_xxx_val = np.zeros((NQ, NC, ldof))
        phi_yxx_val = np.zeros((NQ, NC, ldof))
        phi_y_val = np.zeros((NQ, NC, ldof))
        phi_yy_val = np.zeros((NQ, NC, ldof))
        phi_yyy_val = np.zeros((NQ, NC, ldof))
        phi_xyy_val = np.zeros((NQ, NC, ldof))

        # # 替换成 numpy 可计算的函数
        for n in range(ldof):
            func = lambdify([self.lam0, self.lam1, self.lam2], phi[n], 'numpy')
            phi_val[:, n] = func(bcs[:, 0], bcs[:, 1], bcs[:, 2])

            func = lambdify([self.lam0, self.lam1, self.lam2, self.lam0_x, self.lam0_y, self.lam1_x, self.lam1_y,
                             self.lam2_x, self.lam2_y], phi_x[n], 'numpy')
            phi_x_val[..., n] = func(bcs[:, 0].reshape(NQ, 1), bcs[:, 1].reshape(NQ, 1), bcs[:, 2].reshape(NQ, 1),
                                     Dlambda[:, 0, 0], Dlambda[:, 0, 1],
                                     Dlambda[:, 1, 0], Dlambda[:, 1, 1],
                                     Dlambda[:, 2, 0], Dlambda[:, 2, 1])

            func = lambdify([self.lam0, self.lam1, self.lam2, self.lam0_x, self.lam0_y, self.lam1_x, self.lam1_y,
                             self.lam2_x, self.lam2_y], phi_xx[n], 'numpy')
            phi_xx_val[..., n] = func(bcs[:, 0].reshape(NQ, 1), bcs[:, 1].reshape(NQ, 1), bcs[:, 2].reshape(NQ, 1),
                                     Dlambda[:, 0, 0], Dlambda[:, 0, 1],
                                     Dlambda[:, 1, 0], Dlambda[:, 1, 1],
                                     Dlambda[:, 2, 0], Dlambda[:, 2, 1])

            func = lambdify([self.lam0, self.lam1, self.lam2, self.lam0_x, self.lam0_y, self.lam1_x, self.lam1_y,
                             self.lam2_x, self.lam2_y], phi_xxx[n], 'numpy')
            phi_xxx_val[..., n] = func(bcs[:, 0].reshape(NQ, 1), bcs[:, 1].reshape(NQ, 1), bcs[:, 2].reshape(NQ, 1),
                                     Dlambda[:, 0, 0], Dlambda[:, 0, 1],
                                     Dlambda[:, 1, 0], Dlambda[:, 1, 1],
                                     Dlambda[:, 2, 0], Dlambda[:, 2, 1])

            func = lambdify([self.lam0, self.lam1, self.lam2, self.lam0_x, self.lam0_y, self.lam1_x, self.lam1_y,
                             self.lam2_x, self.lam2_y], phi_yxx[n], 'numpy')
            phi_yxx_val[..., n] = func(bcs[:, 0].reshape(NQ, 1), bcs[:, 1].reshape(NQ, 1), bcs[:, 2].reshape(NQ, 1),
                                     Dlambda[:, 0, 0], Dlambda[:, 0, 1],
                                     Dlambda[:, 1, 0], Dlambda[:, 1, 1],
                                     Dlambda[:, 2, 0], Dlambda[:, 2, 1])

            func = lambdify([self.lam0, self.lam1, self.lam2, self.lam0_x, self.lam0_y, self.lam1_x, self.lam1_y,
                             self.lam2_x, self.lam2_y], phi_y[n], 'numpy')
            phi_y_val[..., n] = func(bcs[:, 0].reshape(NQ, 1), bcs[:, 1].reshape(NQ, 1), bcs[:, 2].reshape(NQ, 1),
                                     Dlambda[:, 0, 0], Dlambda[:, 0, 1],
                                     Dlambda[:, 1, 0], Dlambda[:, 1, 1],
                                     Dlambda[:, 2, 0], Dlambda[:, 2, 1])

            func = lambdify([self.lam0, self.lam1, self.lam2, self.lam0_x, self.lam0_y, self.lam1_x, self.lam1_y,
                             self.lam2_x, self.lam2_y], phi_yy[n], 'numpy')
            phi_yy_val[..., n] = func(bcs[:, 0].reshape(NQ, 1), bcs[:, 1].reshape(NQ, 1), bcs[:, 2].reshape(NQ, 1),
                                     Dlambda[:, 0, 0], Dlambda[:, 0, 1],
                                     Dlambda[:, 1, 0], Dlambda[:, 1, 1],
                                     Dlambda[:, 2, 0], Dlambda[:, 2, 1])

            func = lambdify([self.lam0, self.lam1, self.lam2, self.lam0_x, self.lam0_y, self.lam1_x, self.lam1_y,
                             self.lam2_x, self.lam2_y], phi_yyy[n], 'numpy')
            phi_yyy_val[..., n] = func(bcs[:, 0].reshape(NQ, 1), bcs[:, 1].reshape(NQ, 1), bcs[:, 2].reshape(NQ, 1),
                                     Dlambda[:, 0, 0], Dlambda[:, 0, 1],
                                     Dlambda[:, 1, 0], Dlambda[:, 1, 1],
                                     Dlambda[:, 2, 0], Dlambda[:, 2, 1])

            func = lambdify([self.lam0, self.lam1, self.lam2, self.lam0_x, self.lam0_y, self.lam1_x, self.lam1_y,
                             self.lam2_x, self.lam2_y], phi_xyy[n], 'numpy')
            phi_xyy_val[..., n] = func(bcs[:, 0].reshape(NQ, 1), bcs[:, 1].reshape(NQ, 1), bcs[:, 2].reshape(NQ, 1),
                                     Dlambda[:, 0, 0], Dlambda[:, 0, 1],
                                     Dlambda[:, 1, 0], Dlambda[:, 1, 1],
                                     Dlambda[:, 2, 0], Dlambda[:, 2, 1])







# # --- to test --- # #
if __name__ == '__main__':
    NN = 4
    box = [0, 1, 0, 1]
    mesh = MF.boxmesh2d(box, nx=NN, ny=NN, meshtype='tri')
    p = 3
    integralalg = FEMeshIntegralAlg(mesh, p + 4, cellmeasure=mesh.entity_measure('cell'))
    c_q = integralalg.cellintegrator
    c_bcs, c_ws = c_q.get_quadrature_points_and_weights()  # c_bcs.shape: (NQ,GD+1)

    cb = compute_basis(p)
    # phi, phi_x, phi_xx, phi_yxx, phi_y, phi_yy, phi_xyy = cb.get_basis_details()
    # phi0 = phi[0]
    cb.get_highorder_diff(c_bcs, mesh.grad_lambda())



    print('end of the test')





print('end of the file')






