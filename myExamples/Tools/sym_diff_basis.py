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
from fealpy.functionspace import LagrangeFiniteElementSpace


class compute_basis:
    def __init__(self, p, mesh):
        self.p = p
        self.mesh = mesh
        self.lam0, self.lam1, self.lam2 = symbols('lam0 lam1 lam2')
        self.lam0_x, self.lam1_x, self.lam2_x = symbols('lam0_x lam1_x lam2_x')
        self.lam0_y, self.lam1_y, self.lam2_y = symbols('lam0_y lam1_y lam2_y')

    def basis(self, p, m, n, k):
        """
        :param p: 多项式阶数.
        :param m, n, k: fealpy 中的 multi_index_matrix2d 指标.
        ---
        这个函数是通用的, 给定 p, m, n, k 后返回相应基函数的'表达式'.
        """
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
        """
        这个函数本身被设计为 *最多* 只能求 3 阶导数
        """
        basis = self.basis
        p = self.p

        index = multi_index_matrix2d(p)
        n = len(index)
        phi_list = []
        phi_x_list = []
        phi_y_list = []
        phi_xy_list = []
        phi_xx_list = []
        phi_yy_list = []
        phi_xxx_list = []
        phi_yyy_list = []
        phi_yxx_list = []
        phi_xyy_list = []

        for i in range(n):
            phi_list.append(basis(p, index[i, 0], index[i, 1], index[i, 2]))
            phi_x_list.append(diff(phi_list[i], self.lam0) * self.lam0_x
                              + diff(phi_list[i], self.lam1) * self.lam1_x
                              + diff(phi_list[i], self.lam2) * self.lam2_x)
            phi_y_list.append(diff(phi_list[i], self.lam0) * self.lam0_y
                              + diff(phi_list[i], self.lam1) * self.lam1_y
                              + diff(phi_list[i], self.lam2) * self.lam2_y)
            phi_xy_list.append(diff(phi_x_list[i], self.lam0) * self.lam0_y
                               + diff(phi_x_list[i], self.lam1) * self.lam1_y
                               + diff(phi_x_list[i], self.lam2) * self.lam2_y)
            phi_xx_list.append(diff(phi_x_list[i], self.lam0) * self.lam0_x
                               + diff(phi_x_list[i], self.lam1) * self.lam1_x
                               + diff(phi_x_list[i], self.lam2) * self.lam2_x)
            phi_yy_list.append(diff(phi_y_list[i], self.lam0) * self.lam0_y
                               + diff(phi_y_list[i], self.lam1) * self.lam1_y
                               + diff(phi_y_list[i], self.lam2) * self.lam2_y)
            phi_xxx_list.append(diff(phi_xx_list[i], self.lam0) * self.lam0_x
                                + diff(phi_xx_list[i], self.lam1) * self.lam1_x
                                + diff(phi_xx_list[i], self.lam2) * self.lam2_x)
            phi_yyy_list.append(diff(phi_yy_list[i], self.lam0) * self.lam0_y
                                + diff(phi_yy_list[i], self.lam1) * self.lam1_y
                                + diff(phi_yy_list[i], self.lam2) * self.lam2_y)
            phi_yxx_list.append(diff(phi_xx_list[i], self.lam0) * self.lam0_y
                                + diff(phi_xx_list[i], self.lam1) * self.lam1_y
                                + diff(phi_xx_list[i], self.lam2) * self.lam2_y)
            phi_xyy_list.append(diff(phi_yy_list[i], self.lam0) * self.lam0_x
                                + diff(phi_yy_list[i], self.lam1) * self.lam1_x
                                + diff(phi_yy_list[i], self.lam2) * self.lam2_x)
        return phi_list, phi_x_list, phi_y_list, phi_xy_list, phi_xx_list,  phi_yy_list, phi_xxx_list, phi_yyy_list, \
               phi_yxx_list, phi_xyy_list

    def get_highorder_diff(self, bcs, p=None, order='all'):
        """

        :param bcs:
        :param p:
        :param order:
        :return:
        """
        p = self.p if p is None else p
        Dlambda = self.mesh.grad_lambda()

        phi, phi_x, phi_y, phi_xy, phi_xx, phi_yy, phi_xxx, phi_yyy, phi_yxx, phi_xyy = self.diff_basis()
        # basis = self.get_basis_details()
        # Nb = len(basis)
        # ldof = len(basis[0])
        ldof = len(phi)

        NQ = bcs.shape[0]
        NC = Dlambda.shape[0]
        phi_val = np.zeros((NQ, ldof))
        phi_x_val = np.zeros((NQ, NC, ldof))
        phi_y_val = np.zeros((NQ, NC, ldof))
        phi_xy_val = np.zeros((NQ, NC, ldof))
        phi_xx_val = np.zeros((NQ, NC, ldof))
        phi_yy_val = np.zeros((NQ, NC, ldof))
        phi_xxx_val = np.zeros((NQ, NC, ldof))
        phi_yyy_val = np.zeros((NQ, NC, ldof))
        phi_yxx_val = np.zeros((NQ, NC, ldof))
        phi_xyy_val = np.zeros((NQ, NC, ldof))

        # |--- 替换成 numpy 可计算的函数
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
                             self.lam2_x, self.lam2_y], phi_y[n], 'numpy')
            phi_y_val[..., n] = func(bcs[:, 0].reshape(NQ, 1), bcs[:, 1].reshape(NQ, 1), bcs[:, 2].reshape(NQ, 1),
                                     Dlambda[:, 0, 0], Dlambda[:, 0, 1],
                                     Dlambda[:, 1, 0], Dlambda[:, 1, 1],
                                     Dlambda[:, 2, 0], Dlambda[:, 2, 1])

            func = lambdify([self.lam0, self.lam1, self.lam2, self.lam0_x, self.lam0_y, self.lam1_x, self.lam1_y,
                             self.lam2_x, self.lam2_y], phi_xy[n], 'numpy')
            phi_xy_val[..., n] = func(bcs[:, 0].reshape(NQ, 1), bcs[:, 1].reshape(NQ, 1), bcs[:, 2].reshape(NQ, 1),
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
                             self.lam2_x, self.lam2_y], phi_yy[n], 'numpy')
            phi_yy_val[..., n] = func(bcs[:, 0].reshape(NQ, 1), bcs[:, 1].reshape(NQ, 1), bcs[:, 2].reshape(NQ, 1),
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
                             self.lam2_x, self.lam2_y], phi_yyy[n], 'numpy')
            phi_yyy_val[..., n] = func(bcs[:, 0].reshape(NQ, 1), bcs[:, 1].reshape(NQ, 1), bcs[:, 2].reshape(NQ, 1),
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
                             self.lam2_x, self.lam2_y], phi_xyy[n], 'numpy')
            phi_xyy_val[..., n] = func(bcs[:, 0].reshape(NQ, 1), bcs[:, 1].reshape(NQ, 1), bcs[:, 2].reshape(NQ, 1),
                                       Dlambda[:, 0, 0], Dlambda[:, 0, 1],
                                       Dlambda[:, 1, 0], Dlambda[:, 1, 1],
                                       Dlambda[:, 2, 0], Dlambda[:, 2, 1])

        if order in ['3order', '3-order', '3rd-order', 'all-3rd-order']:
            return phi_xxx_val, phi_yyy_val, phi_yxx_val, phi_xyy_val
        elif order in ['2order', '2-order', '2nd-order', 'all-2nd-order']:
            return phi_xy_val, phi_xx_val, phi_yy_val
        elif order in ['1order', '1-order', '1st-order', 'all-1st-order']:
            return phi_x_val, phi_y_val
        else:
            return phi_val[..., np.newaxis, :], phi_x_val, phi_y_val, phi_xy_val, phi_xx_val, phi_yy_val, phi_xxx_val, \
                   phi_yyy_val, phi_yxx_val, phi_xyy_val


# |--- to test ---| #
# if __name__ == '__main__':
#     NN = 2
#     box = [0, 1, 0, 1]
#     mesh = MF.boxmesh2d(box, nx=NN, ny=NN, meshtype='tri')
#     p = 3
#     integralalg = FEMeshIntegralAlg(mesh, p + 2, cellmeasure=mesh.entity_measure('cell'))
#     c_q = integralalg.cellintegrator
#     c_bcs, c_ws = c_q.get_quadrature_points_and_weights()  # c_bcs.shape: (NQ,GD+1)
#
#     cb = compute_basis(p, mesh)
#     phi, phi_x, phi_y, phi_xy, phi_xx, phi_yy, phi_xxx, phi_yyy, phi_yxx, phi_xyy = cb.get_highorder_diff(c_bcs)
#
#     # # Lagrange space
#     space = LagrangeFiniteElementSpace(mesh, p)
#     s_phi = space.basis(c_bcs)  # (NQ,1,fldof)
#     s_gphi = space.grad_basis(c_bcs)  # (NQ,NC,ldof,2)
#     s_gphi0 = s_gphi[..., 0]  # phi_x, (NQ,NC,ldof)
#     s_gphi1 = s_gphi[..., 1]  # phi_y, (NQ,NC,ldof)
#
#     np.allclose(phi_x, s_gphi0)
#     np.allclose(phi_y, s_gphi1)
#
#     print('end of the test')






