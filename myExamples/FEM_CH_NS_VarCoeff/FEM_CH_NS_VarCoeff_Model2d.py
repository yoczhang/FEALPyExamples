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
        self.stressC = 3.

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
        eta_max = max(nu0 / rho0, nu1 / rho1)
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
        uh_val_f = self.space.value(uh, self.f_bcs)[..., self.DirCellIdx_NS]  # (NQ,NDir)
        grad_uh_val = self.space.grad_value(uh, self.c_bcs)  # (NQ,NC,GD)
        vel0_val = self.vspace.value(vel0, self.c_bcs)  # (NQ,NC)
        vel1_val = self.vspace.value(vel1, self.c_bcs)  # (NQ,NC)
        grad_vel0_val = self.vspace.grad_value(vel0, self.c_bcs)  # (NQ,NC,GD)
        grad_vel1_val = self.vspace.grad_value(vel1, self.c_bcs)  # (NQ,NC,GD)

        nolinear_val = self.NSNolinearTerm(vel0, vel1, self.c_bcs)  # nolinear_val.shape: (NQ,NC,GD)
        # nolinear_val0 = nolinear_val[..., 0]  # (NQ,NC)
        # nolinear_val1 = nolinear_val[..., 1]  # (NQ,NC)

        velDir_val = self.pde.dirichlet_NS(self.f_pp_Dir_NS, next_t)  # (NQ,NDir,GD)
        f_val_NS = self.pde.source_NS(self.c_pp, next_t, self.pde.epsilon, self.pde.eta, m, rho0, rho1, nu0, nu1, self.stressC)  # (NQ,NC,GD)
        # Neumann_0 = self.pde.neumann_0_NS(self.f_pp_Neu_NS, next_t, self.nNeu_NS)  # (NQ,NE)
        # Neumann_1 = self.pde.neumann_1_NS(self.f_pp_Neu_NS, next_t, self.nNeu_NS)  # (NQ,NE)

        # --- the CH_term: uh_val * (-epsilon*\nabla\Delta uh_val + \nabla free_energy)
        grad_free_energy_c = self.pde.epsilon / self.pde.eta ** 2 * self.grad_free_energy_at_cells(uh, self.c_bcs)  # (NQ,NC,2)
        if self.p < 3:
            grad_x_laplace_uh = np.zeros(grad_free_energy_c[..., 0].shape)
            grad_y_laplace_uh = np.zeros(grad_free_energy_c[..., 0].shape)
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
        rho_n_f = (rho0 + rho1) / 2. + (rho0 - rho1) / 2. * uh_val_f  # (NQ,NDir)
        nu_n = (nu0 + nu1) / 2. + (nu0 - nu1) / 2. * uh_val  # (NQ,NC)
        nu_n_f = (nu0 + nu1) / 2. + (nu0 - nu1) / 2. * uh_val_f  # (NQ,NDir)
        J_n0 = J0 * (grad_x_laplace_uh + grad_free_energy_c[..., 0])  # (NQ,NC)
        J_n1 = J0 * (grad_y_laplace_uh + grad_free_energy_c[..., 1])  # (NQ,NC)

        # --- the auxiliary variable: G_VC
        stressC = self.stressC
        vel_stress_mat = [[stressC*grad_vel0_val[..., 0], stressC*0.5 * (grad_vel0_val[..., 1] + grad_vel1_val[..., 0])],
                          [stressC*0.5 * (grad_vel0_val[..., 1] + grad_vel1_val[..., 0]), stressC*grad_vel1_val[..., 1]]]
        vel_grad_mat = [[grad_vel0_val[..., 0], grad_vel0_val[..., 1]],
                        [grad_vel1_val[..., 0], grad_vel1_val[..., 1]]]
        rho_n_axis = rho_n[..., np.newaxis]
        G_VC = (-nolinear_val + (1. / rho_min - 1. / rho_n_axis) * grad_ph_val
                + 1. / rho_n_axis * self.vec_div_mat((nu0 - nu1) / 2. * grad_uh_val, vel_stress_mat)
                - 1. / rho_n_axis * np.array([CH_term_val0, CH_term_val1]).transpose((1, 2, 0))
                - 1. / rho_n_axis * self.vec_div_mat([J_n0, J_n1], vel_grad_mat) + 1. / rho_n_axis * f_val_NS
                + 1./self.dt * np.array([vel0_val, vel1_val]).transpose((1, 2, 0)))  # (NQ,NC,2)

        # # >>>>>>> test1 >>>>>>>
        # tt0 = self.vec_div_mat((nu0 - nu1) / 2. * grad_uh_val, vel_stress_mat)
        #
        # stressC_1 = 1.
        # vel_stress_mat_1 = [
        #     [stressC_1 * grad_vel0_val[..., 0], stressC_1 * 0.5 * (grad_vel0_val[..., 1] + grad_vel1_val[..., 0])],
        #     [stressC_1 * 0.5 * (grad_vel0_val[..., 1] + grad_vel1_val[..., 0]), stressC_1 * grad_vel1_val[..., 1]]]
        # tt1 = 2.*self.vec_div_mat((nu0 - nu1) / 2. * grad_uh_val, vel_stress_mat_1)
        # # <<<<<<< test1 <<<<<<<

        # # >>>>>>> test2 >>>>>>>
        stressC_1 = 1.
        nu0 = 2.*nu0
        nu1 = 2.*nu1
        f_val_NS_1 = self.pde.source_NS(self.c_pp, next_t, self.pde.epsilon, self.pde.eta, m, rho0, rho1, nu0, nu1, stressC_1)  # (NQ,NC,GD)
        vel_stress_mat_1 = [
            [stressC_1 * grad_vel0_val[..., 0], stressC_1 * 0.5 * (grad_vel0_val[..., 1] + grad_vel1_val[..., 0])],
            [stressC_1 * 0.5 * (grad_vel0_val[..., 1] + grad_vel1_val[..., 0]), stressC_1 * grad_vel1_val[..., 1]]]
        G_VC_1 = (-nolinear_val + (1. / rho_min - 1. / rho_n_axis) * grad_ph_val
                  + 1. / rho_n_axis * self.vec_div_mat((nu0 - nu1) / 2. * grad_uh_val, vel_stress_mat_1)
                  - 1. / rho_n_axis * np.array([CH_term_val0, CH_term_val1]).transpose((1, 2, 0))
                  - 1. / rho_n_axis * self.vec_div_mat([J_n0, J_n1], vel_grad_mat) + 1. / rho_n_axis * f_val_NS_1
                  + 1. / self.dt * np.array([vel0_val, vel1_val]).transpose((1, 2, 0)))  # (NQ,NC,2)

        if np.allclose(G_VC_1, G_VC):
            print('G_VC_1 == G_VC\n')
        else:
            raise ValueError("G_VC_1 != G_VC")
        # # <<<<<<< test2 <<<<<<<

        eta_n = nu_n / rho_n  # (NQ,NC)
        eta_n_f = nu_n_f / rho_n_f  # (NQ,NDir)
        eta_nx = ((nu0 - nu1) / 2. * rho_n - (rho0 - rho1) / 2. * nu_n) * grad_uh_val[..., 0] / rho_n ** 2  # (NQ,NC)
        eta_ny = ((nu0 - nu1) / 2. * rho_n - (rho0 - rho1) / 2. * nu_n) * grad_uh_val[..., 1] / rho_n ** 2  # (NQ,NC)
        curl_vel = grad_vel1_val[..., 0] - grad_vel0_val[..., 1]  # (NQ,NC)
        curl_vel_f = grad_vel1_f[..., 0] - grad_vel0_f[..., 1]  # (NQ,NDir)
        # n_curl_curl_vel_f = np.array([nDir_NS[:, 1]*curl_vel_f, -nDir_NS[:, 0]*curl_vel_f]).transpose((1, 2, 0))  # (NQ,NDir,GD)
        # grad_ratio_curl_curl_vel = 1

        # for Dirichlet faces integration
        dir_int0 = -1 / self.dt * np.einsum('i, ijk, jk, ijn, j->jn', self.f_ws, velDir_val, nDir_NS, self.phi_f,
                                            self.DirEdgeMeasure_NS)  # (NDir,fldof)
        dir_int1 = -(np.einsum('i, j, ij, jin, j->jn', self.f_ws, nDir_NS[:, 1], eta_n_f*curl_vel_f, self.gphi_f[..., 0],
                               self.DirEdgeMeasure_NS)
                     + np.einsum('i, j, ij, jin, j->jn', self.f_ws, -nDir_NS[:, 0], eta_n_f*curl_vel_f, self.gphi_f[..., 1],
                                 self.DirEdgeMeasure_NS))  # (NDir,cldof)

        # for cell integration
        cell_int0 = np.einsum('i, ijs, ijks, j->jk', self.c_ws, G_VC, self.gphi_c, self.cellmeasure)  # (NC,cldof)
        cell_int1 = (np.einsum('i, ij, ijk, j->jk', self.c_ws, eta_ny * curl_vel, self.gphi_c[..., 0], self.cellmeasure)
                     + np.einsum('i, ij, ijk, j->jk', self.c_ws, -eta_nx * curl_vel, self.gphi_c[..., 1], self.cellmeasure))  # (NC,cldof)

        # # --- 1. assemble the NS's pressure equation
        prv = np.zeros((self.dof.number_of_global_dofs(),), dtype=self.ftype)  # (Npdof,) the Pressure's Right-hand Vector
        np.add.at(prv, self.face2dof[self.DirEdgeIdx_NS, :], dir_int0)
        np.add.at(prv, self.cell2dof[self.DirCellIdx_NS, :], dir_int1)
        np.add.at(prv, self.cell2dof, cell_int0 + cell_int1)

        # # --- 2. solve the NS's pressure equation
        plsm = 1. / rho_min * self.StiffMatrix

        # # Method I: The following code is right! Pressure satisfies \int_\Omega p = 0
        basis_int = self.space.integral_basis()
        plsm_temp = bmat([[plsm, basis_int.reshape(-1, 1)], [basis_int, None]], format='csr')
        prv = np.r_[prv, 0]
        ph[:] = spsolve(plsm_temp, prv)[:-1]  # we have added one addtional dof
        # ph[:] = spsolve(plsm, prv)

        # # Method II: Using the Dirichlet boundary of pressure
        # def dir_pressure(p):
        #     return self.pde.pressure_NS(p, next_t)
        # bc = DirichletBC(self.space, dir_pressure)
        # plsm_temp, prv = bc.apply(plsm.copy(), prv)
        # ph[:] = spsolve(plsm_temp, prv).reshape(-1)

        # # ------------------------------------ # #
        # # --- to update the velocity value --- # #
        # # ------------------------------------ # #
        grad_ph = self.space.grad_value(ph, self.c_bcs)  # (NQ,NC,2)

        # # the Velocity-Left-Matrix
        vlm0 = 1 / self.dt * self.vel_MM + eta_max * self.vel_SM
        vlm1 = vlm0.copy()

        # # to get the u's Right-hand Vector
        def dir_u0(p):
            return self.pde.dirichlet_NS(p, next_t)[..., 0]

        def dir_u1(p):
            return self.pde.dirichlet_NS(p, next_t)[..., 1]

        # # --- assemble the first-component of Velocity-Right-Vector
        vrv0 = np.zeros((self.vdof.number_of_global_dofs(),), dtype=self.ftype)  # (Nvdof,)
        vrv0_c_0 = np.einsum('i, ij, ijk, j->jk', self.c_ws, - 1. / rho_min * grad_ph[..., 0] + G_VC[..., 0], self.vphi_c,
                             self.cellmeasure)  # (NC,clodf)
        vrv0_c_1 = (np.einsum('i, ij, ijk, j->jk', self.c_ws, (eta_n - eta_max) * curl_vel, self.vgphi_c[..., 1], self.cellmeasure)
                    + np.einsum('i, ij, ijk, j->jk', self.c_ws, eta_ny * curl_vel, self.vphi_c, self.cellmeasure))  # (NC,clodf)
        np.add.at(vrv0, self.vcell2dof, vrv0_c_0 + vrv0_c_1)

        v0_bc = DirichletBC(self.vspace, dir_u0, threshold=self.DirEdgeIdx_NS)
        vlm0, vrv0 = v0_bc.apply(vlm0, vrv0)
        vel0[:] = spsolve(vlm0, vrv0).reshape(-1)

        # # --- assemble the second-component of Velocity-Right-Vector
        vrv1 = np.zeros((self.vdof.number_of_global_dofs(),), dtype=self.ftype)  # (Nvdof,)
        vrv1_c_0 = np.einsum('i, ij, ijk, j->jk', self.c_ws, - 1. / rho_min * grad_ph[..., 1] + G_VC[..., 1], self.vphi_c,
                             self.cellmeasure)  # (NC,clodf)
        vrv1_c_1 = (np.einsum('i, ij, ijk, j->jk', self.c_ws, (eta_n - eta_max) * curl_vel, -self.vgphi_c[..., 0], self.cellmeasure)
                    + np.einsum('i, ij, ijk, j->jk', self.c_ws, - eta_nx * curl_vel, self.vphi_c, self.cellmeasure))  # (NC,clodf)
        np.add.at(vrv1, self.vcell2dof, vrv1_c_0 + vrv1_c_1)

        v1_bc = DirichletBC(self.vspace, dir_u1, threshold=self.DirEdgeIdx_NS)
        vlm1, vrv0 = v1_bc.apply(vlm1, vrv1)
        vel1[:] = spsolve(vlm1, vrv1).reshape(-1)

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
