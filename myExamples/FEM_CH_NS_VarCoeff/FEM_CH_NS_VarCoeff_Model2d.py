#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: FEM_CH_NS_VarCoeff_Model2d.py
# @Author: Yongchao Zhang, Northwest University, Xi'an
# @E-mail: yoczhang@nwu.edu.cn
# @Site:
# @Time: Dec 13, 2021
# ---

__doc__ = """
The FEM for Variable-Coefficient coupled Cahn-Hilliard-Navier-Stokes model in 2D. 
"""

import numpy as np
from scipy.sparse import csr_matrix, spdiags, eye, bmat
from fealpy.quadrature import FEMeshIntegralAlg
from scipy.sparse.linalg import spsolve
from fealpy.boundarycondition import DirichletBC
from fealpy.decorator import timer
from fealpy.functionspace import LagrangeFiniteElementSpace
from sym_diff_basis import compute_basis
from FEM_CH_NS_Model2d import FEM_CH_NS_Model2d


class FEM_CH_NS_VarCoeff_Model2d(FEM_CH_NS_Model2d):
    def __init__(self, pde, mesh, p, dt):
        super(FEM_CH_NS_VarCoeff_Model2d, self).__init__(pde, mesh, p, dt)

    def decoupled_NS_Solver_T1stOrder(self, vel0, vel1, ph, uh, next_t):
        """
        The decoupled-Navier-Stokes-solver for the all system.
        :param vel0: The fist-component of velocity: stored the n-th(time) value, and to update the (n+1)-th value.
        :param vel1: The second-component of velocity: stored the n-th(time) value, and to update the (n+1)-th value.
        :param ph: The pressure: stored the n-th(time) value, and to update the (n+1)-th value.
        :param uh: The n-th(time) value of the solution of Cahn-Hilliard equation.
        :param next_t: The next-time.
        :return: Updated vel0, vel1, ph.
        """

        pde = self.pde
        # # variable coefficients settings
        m = pde.m
        rho0 = pde.rho0
        rho1 = pde.rho1
        nu0 = pde.nu0
        nu1 = pde.nu1

        rho_min = min(rho0, rho1)
        ratio_max = max(nu0 / rho0, nu1 / rho1)
        J0 = -1. / 2 * (rho0 - rho1) * m

        nDir_NS = self.nDir_NS  # (NDir,GD), here, the GD is 2

        # # the pre-settings
        grad_vel0_f = self.uh_grad_value_at_faces(vel0, self.f_bcs, self.DirCellIdx_NS, self.DirLocalIdx_NS,
                                                  space=self.vspace)  # grad_vel0: (NQ,NDir,GD)
        grad_vel1_f = self.uh_grad_value_at_faces(vel1, self.f_bcs, self.DirCellIdx_NS, self.DirLocalIdx_NS,
                                                  space=self.vspace)  # grad_vel1: (NQ,NDir,GD)

        # for cell-integration
        grad_ph_val = self.space.grad_value(ph, self.c_bcs)  # (NQ,NC,GD)
        uh_val = self.space.value(uh, self.c_bcs)  # (NQ,NC)
        grad_uh_val = self.space.grad_value(uh, self.c_bcs)  # (NQ,NC,GD)
        vel0_val = self.vspace.value(vel0, self.c_bcs)  # (NQ,NC)
        vel1_val = self.vspace.value(vel1, self.c_bcs)  # (NQ,NC)
        grad_vel0_val = self.vspace.grad_value(vel0, self.c_bcs)  # (NQ,NC,GD)
        grad_vel1_val = self.vspace.grad_value(vel1, self.c_bcs)  # (NQ,NC,GD)

        nolinear_val = self.NSNolinearTerm(vel0, vel1, self.c_bcs)  # nolinear_val.shape: (NQ,NC,GD)
        nolinear_val0 = nolinear_val[..., 0]  # (NQ,NC)
        nolinear_val1 = nolinear_val[..., 1]  # (NQ,NC)

        velDir_val = self.pde.dirichlet_NS(self.f_pp_Dir_NS, next_t)  # (NQ,NDir,GD)
        f_val_NS = self.pde.source_NS(self.c_pp, next_t, self.pde.nu, self.pde.epsilon, self.pde.eta)  # (NQ,NC,GD)
        Neumann_0 = self.pde.neumann_0_NS(self.f_pp_Neu_NS, next_t, self.nNeu_NS)  # (NQ,NE)
        Neumann_1 = self.pde.neumann_1_NS(self.f_pp_Neu_NS, next_t, self.nNeu_NS)  # (NQ,NE)

        # --- the CH_term: -epsilon*\nabla\Delta uh_val + \nabla free_energy
        grad_free_energy_c = self.pde.epsilon / self.pde.eta ** 2 * self.grad_free_energy_at_cells(uh, self.c_bcs)  # (NQ,NC,2)
        if self.p < 3:
            CH_term_val0 = uh_val * grad_free_energy_c[..., 0]  # (NQ,NC)
            CH_term_val1 = uh_val * grad_free_energy_c[..., 1]  # (NQ,NC)
        elif self.p == 3:
            phi_xxx, phi_yyy, phi_yxx, phi_xyy = self.cb.get_highorder_diff(self.c_bcs, order='3rd-order')  # (NQ,NC,ldof)
            grad_x_laplace_uh = -self.pde.epsilon * np.einsum('ijk, jk->ij', phi_xxx + phi_xyy, uh[self.cell2dof])  # (NQ,NC)
            grad_y_laplace_uh = -self.pde.epsilon * np.einsum('ijk, jk->ij', phi_yxx + phi_yyy, uh[self.cell2dof])  # (NQ,NC)
            CH_term_val0 = uh_val * (grad_x_laplace_uh + grad_free_energy_c[..., 0])  # (NQ,NC)
            CH_term_val1 = uh_val * (grad_y_laplace_uh + grad_free_energy_c[..., 1])  # (NQ,NC)
        else:
            raise ValueError("The polynomial order p should be <= 3.")

        # --- update the variable coefficients
        rho_n = (rho0 + rho1) / 2. + (rho0 - rho1) / 2. * uh_val  # (NQ,NC)
        nu_n = (nu0 + nu1) / 2. + (nu0 + nu1) / 2. * uh_val  # (NQ,NC)
        J_n0 = J0 * CH_term_val0  # (NQ,NC)
        J_n1 = J0 * CH_term_val1  # (NQ,NC)

        # --- the auxiliary variable: G_VC
        stress_mat = [[grad_vel0_val[..., 0], 0.5 * (grad_vel0_val[..., 1] + grad_vel1_val[..., 0])],
                      [0.5 * (grad_vel0_val[..., 1] + grad_vel1_val[..., 0]), grad_vel1_val[..., 1]]]
        grad_vel_mat = [[grad_vel0_val[..., 0], grad_vel0_val[..., 1]],
                        [grad_vel1_val[..., 0], grad_vel1_val[..., 1]]]
        # G_VC = np.empty(nolinear_val.shape, dtype=self.ftype)  # G_VC.shape: (NQ,NC,2)
        rho_n_axis = rho_n[..., np.newaxis]
        G_VC = -nolinear_val + (1. / rho0 - 1. / rho_n_axis) * grad_ph_val \
               + 1. / rho_n_axis * self.vec_div_mat((nu0 - nu1) / 2. * grad_uh_val, stress_mat) \
               - 1. / rho_n_axis * np.array([CH_term_val0, CH_term_val1]).transpose((1, 2, 0)) \
               - 1. / rho_n_axis * self.vec_div_mat([J_n0, J_n1], grad_vel_mat) + 1. / rho_n_axis * f_val_NS \
               + 1./self.dt * np.array([vel0_val, vel1_val]).transpose((1, 2, 0))  # (NQ,NC,2)

        ratio_n = nu_n / rho_n  # (NQ,NC)
        ratio_nx = ((nu0 - nu1) / 2. * rho_n - (rho0 - rho1) / 2. * nu_n) * grad_uh_val[..., 0] / rho_n ** 2  # (NQ,NC)
        ratio_ny = ((nu0 - nu1) / 2. * rho_n - (rho0 - rho1) / 2. * nu_n) * grad_uh_val[..., 1] / rho_n ** 2  # (NQ,NC)
        curl_vel_f = grad_vel1_f[..., 0] - grad_vel0_f[..., 1]  # (NQ,NDir)
        curl_vel = grad_vel1_val[..., 0] - grad_vel0_val[..., 1]  # (NQ,NC)
        n_curl_curl_vel_f = np.array([nDir_NS[:, 1]*curl_vel_f, -nDir_NS[:, 0]*curl_vel_f]).transpose((1, 2, 0))  # (NQ,NDir,GD)

        # for Dirichlet faces integration
        dir_int0 = -1 / self.dt * np.einsum('i, ijk, jk, ijn, j->jn', self.f_ws, velDir_val, self.nDir_NS, self.phi_f,
                                            self.DirEdgeMeasure_NS)  # (NDir,fldof)



    def vec_div_mat(self, vector, matrix):

        if type(vector) is np.ndarray:
            vector = [vector[..., 0], vector[..., 1]]

        val0 = vector[0] * matrix[0][0] + vector[1] * matrix[0][1]
        val1 = vector[0] * matrix[1][0] + vector[1] * matrix[1][1]
        return np.array([val0, val1]).transpose((1, 2, 0))  # (NQ,NC,2)

    def set_CH_Neumann_edge(self, idxNeuEdge=None):
        """
        We overload the `set_CH_Neumann_edge` function, so that (in the parent class) to call this function.
        :param idxNeuEdge:
        :return:
        """
        if idxNeuEdge is not None:
            return idxNeuEdge

        mesh = self.mesh
        edge2cell = mesh.ds.edge_to_cell()
        bdEdge = (edge2cell[:, 0] == edge2cell[:, 1])  # the bool vars, to get the boundary edges

        isNeuEdge = bdEdge  # here, we first set all the boundary edges are Neu edges
        idxNeuEdge, = np.nonzero(isNeuEdge)  # (NE_Dir,)
        return idxNeuEdge

    def set_NS_Dirichlet_edge(self, idxDirEdge=None):
        """
        We overload the `set_NS_Dirichlet_edge` function, so that (in the parent class) to call this function.
        :param idxDirEdge:
        :return:
        """
        if idxDirEdge is not None:
            return idxDirEdge

        mesh = self.mesh
        edge2cell = mesh.ds.edge_to_cell()
        isBdEdge = (edge2cell[:, 0] == edge2cell[:, 1])  # (NE,), the bool vars, to get the boundary edges

        isDirEdge = isBdEdge  # here, we set all the boundary edges are Dir edges
        idxDirEdge, = np.nonzero(isDirEdge)  # (NE_Dir,)

        return idxDirEdge
