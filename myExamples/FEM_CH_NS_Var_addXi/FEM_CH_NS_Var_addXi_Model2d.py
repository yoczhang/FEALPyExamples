#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: FEM_CH_NS_Var_addXi_Model2d.py
# @Author: Yongchao Zhang
# @Institution: Northwest University, Xi'an, Shaanxi, China
# @E-mail: yoczhang@126.com, yoczhang@nwu.edu.cn
# @Site: 
# @Time: Jan 09, 2022
# ---


__doc__ = """
The FEM for Variable-Coefficient coupled Cahn-Hilliard-Navier-Stokes model in 2D,
add the solver for \\xi.
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


class FEM_CH_NS_Var_addXi_Model2d(FEM_CH_NS_Model2d):
    """
    注意:
        本程序所参考的数值格式是按照 $D(u)=\nabla u + \nabla u^T$ 来写的,
        如果要调整为 $D(u)=(\nabla u + \nabla u^T)/2$, 需要调整的地方太多了,
        所以为了保证和数值格式的一致性, 本程序也是严格按照 $D(u)=\nabla u + \nabla u^T$ 来编写的.
    """
    def __init__(self, pde, mesh, p, dt):
        super(FEM_CH_NS_Var_addXi_Model2d, self).__init__(pde, mesh, p, dt)

    def CH_NS_addXi_Solver_T1stOrder(self):
        pde = self.pde
        timemesh = self.timemesh
        NT = len(timemesh)
        dt = self.dt
        uh = self.uh
        wh = self.wh
        # wh_part0 = wh.copy()
        # wh_part1 = wh.copy()
        vel0 = self.vel0
        vel1 = self.vel1
        ph = self.ph

        print('    # #################################### #')
        print('      Time 1st-order scheme')
        print('    # #################################### #')

        print('    # ------------ parameters ------------ #')
        print('    s = %.4e,  alpha = %.4e,  m = %.4e,  epsilon = %.4e,  eta = %.4e' % (self.s, self.alpha, self.pde.m,
                                                                                        self.pde.epsilon, self.pde.eta))
        print('    t0 = %.4e,  T = %.4e, dt = %.4e' % (timemesh[0], timemesh[-1], dt))
        print(' ')

        if pde.haveTrueSolution:
            def init_solution_CH(p):
                return pde.solution_CH(p, 0)
            uh[:] = self.space.interpolation(init_solution_CH)

            def init_velocity0(p):
                return pde.velocity_NS(p, 0)[..., 0]
            vel0[:] = self.vspace.interpolation(init_velocity0)

            def init_velocity1(p):
                return pde.velocity_NS(p, 0)[..., 1]
            vel1[:] = self.vspace.interpolation(init_velocity1)

            def init_pressure(p):
                return pde.pressure_NS(p, 0)
            ph[:] = self.space.interpolation(init_pressure)

        # # time-looping
        print('    # ------------ begin the time-looping ------------ #')
        for nt in range(NT - 1):
            currt_t = timemesh[nt]
            next_t = currt_t + dt

            # # --- decoupled solvers, updated the discrete-solutions to the next-time
            uh_currt = uh.copy()
            # print('        |___ decoupled Cahn-Hilliard Solver(Time-1st-order): ')
            uh_part0, uh_part1, wh_part0, wh_part1 = self.decoupled_CH_addXi_Solver_T1stOrder(uh, wh, vel0, vel1, next_t)
            # print('        -----------------------------------------------')
            # print('        |___ decoupled Navier-Stokes Solver(Time-1st-order): ')
            # self.decoupled_NS_addXi_Solver_T1stOrder(vel0, vel1, ph, uh_currt, next_t)
            # print('    end of one-looping')

            if nt % max([int(NT / 5), 1]) == 0:
                print('    currt_t = %.4e' % currt_t)
                uh_l2err, uh_h1err, vel_l2err, vel_h1err, ph_l2err = self.currt_error(uh, vel0, vel1, ph, timemesh[nt])
                if np.isnan(uh_l2err) | np.isnan(uh_h1err) | np.isnan(vel_l2err) | np.isnan(vel_h1err) | np.isnan(
                        ph_l2err):
                    print('Some error is nan: breaking the program')
                    break
        print('    # ------------ end the time-looping ------------ #\n')

        # # --- errors
        uh_l2err, uh_h1err, vel_l2err, vel_h1err, ph_l2err = self.currt_error(uh, vel0, vel1, ph, timemesh[-1])
        print('    # ------------ the last errors ------------ #')
        print('    uh_l2err = %.4e, uh_h1err = %.4e' % (uh_l2err, uh_h1err))
        print('    vel_l2err = %.4e, vel_h1err = %.4e, ph_l2err = %.4e' % (vel_l2err, vel_h1err, ph_l2err))

        return uh_l2err, uh_h1err, vel_l2err, vel_h1err, ph_l2err

    def decoupled_CH_addXi_Solver_T1stOrder(self, uh, wh, vel0, vel1, next_t):
        """
        The decoupled-Cahn-Hilliard-solver for the all system.
        :param uh: The value of the solution 'phi' of Cahn-Hilliard equation: stored the n-th(time) value, and to update the (n+1)-th value.
        :param wh: The value of the auxiliary solution of Cahn-Hilliard equation: stored the n-th(time) value, and to update the (n+1)-th value.
        :param vel0: The fist-component of NS's velocity: stored the n-th(time) value.
        :param vel1: The second-component of NS's velocity: stored the n-th(time) value.
        :param next_t: Next time.
        :return: Updated uh, wh.
        """

        grad_free_energy_c = self.pde.epsilon / self.pde.eta ** 2 * self.grad_free_energy_at_cells(uh, self.c_bcs)  # (NQ,NC,2)
        grad_free_energy_f = self.pde.epsilon / self.pde.eta ** 2 * \
                             self.grad_free_energy_at_faces(uh, self.f_bcs, self.NeuEdgeIdx_CH, self.NeuCellIdx_CH,
                                                            self.NeuLocalIdx_CH)  # (NQ,NE,2)
        uh_val = self.space.value(uh, self.c_bcs)  # (NQ,NC)
        guh_val_c = self.space.grad_value(uh, self.c_bcs)  # (NQ,NC,2)
        guh_val_f = self.uh_grad_value_at_faces(uh, self.f_bcs, self.NeuCellIdx_CH, self.NeuLocalIdx_CH)  # (NQ,NNeu,2)

        Neumann = self.pde.neumann_CH(self.f_pp_Neu_CH, next_t, self.nNeu_CH)  # (NQ,NE)
        LaplaceNeumann = self.pde.laplace_neumann_CH(self.f_pp_Neu_CH, next_t, self.nNeu_CH)  # (NQ,NE)
        f_val_CH = self.pde.source_CH(self.c_pp, next_t, self.pde.m, self.pde.epsilon, self.pde.eta)  # (NQ,NC)

        # # get the auxiliary equation Right-hand-side-Vector
        aux_rv_part0 = np.zeros((self.dof.number_of_global_dofs(),), dtype=self.ftype)  # (Ndof,)
        aux_rv_part1 = np.zeros((self.dof.number_of_global_dofs(),), dtype=self.ftype)  # (Ndof,)

        # # aux_rhs_c_0:  -1. / (epsilon * m * dt) * (uh^n,phi)_\Omega
        aux_rhs_c_0 = (-1. / (self.pde.epsilon * self.pde.m) *
                       (1 / self.dt * np.einsum('i, ij, ijk, j->jk', self.c_ws, uh_val, self.phi_c, self.cellmeasure)
                        + np.einsum('i, ij, ijk, j->jk', self.c_ws, f_val_CH, self.phi_c, self.cellmeasure)))  # (NC,cldof)

        # # aux_rhs_c_1: -s / epsilon * (\nabla uh^n, \nabla phi)_\Omega
        aux_rhs_c_1 = -self.s / self.pde.epsilon * (
                np.einsum('i, ijm, ijkm, j->jk', self.c_ws, guh_val_c, self.gphi_c, self.cellmeasure))  # (NC,cldof)

        # # aux_rhs_c_2: 1 / epsilon * (\nabla h(uh^n), \nabla phi)_\Omega
        aux_rhs_c_2 = 1. / self.pde.epsilon * (np.einsum('i, ijm, ijkm, j->jk', self.c_ws, grad_free_energy_c,
                                                         self.gphi_c, self.cellmeasure))  # (NC,cldof)

        # # aux_rhs_f_0: (\nabla wh^{n+1}\cdot n, phi)_\Gamma, wh is the solution of auxiliary equation
        # TODO: aux_rhs_f_0 = np.einsum('i, ij, ijn, j->jn', self.f_ws, self.alpha * Neumann + LaplaceNeumann, self.phi_f, self.NeuEdgeMeasure_CH)  # (Nneu,fldof)
        # TODO: this is the boundary condition for the wh (aux-variable), but I DON'T known add to which equation?

        # # aux_rhs_f_1: s / epsilon * (\nabla uh^n \cdot n, phi)_\Gamma
        aux_rhs_f_1 = self.s / self.pde.epsilon * np.einsum('i, ijk, jk, ijn, j->jn', self.f_ws, guh_val_f, self.nNeu_CH,
                                                            self.phi_f, self.NeuEdgeMeasure_CH)  # (Nneu,fldof)

        # # aux_rhs_f_2: -1 / epsilon * (\nabla h(uh^n) \cdot n, phi)_\Gamma
        aux_rhs_f_2 = -1. / self.pde.epsilon * np.einsum('i, ijk, jk, ijn, j->jn', self.f_ws, grad_free_energy_f, self.nNeu_CH,
                                                         self.phi_f, self.NeuEdgeMeasure_CH)  # (Nneu,fldof)

        # # --- now, we add the NS term
        vel0_val_c = self.vspace.value(vel0, self.c_bcs)  # (NQ,NC)
        vel1_val_c = self.vspace.value(vel1, self.c_bcs)  # (NQ,NC)
        vel0_val_f = self.vspace.value(vel0, self.f_bcs)[..., self.bdIndx]  # (NQ,NBE)
        vel1_val_f = self.vspace.value(vel1, self.f_bcs)[..., self.bdIndx]  # (NQ,NBE)

        uh_val_f = self.space.value(uh, self.f_bcs)[..., self.bdIndx]  # (NQ,NBE)
        uh_vel_val_f = np.concatenate([(uh_val_f * vel0_val_f)[..., np.newaxis],
                                       (uh_val_f * vel1_val_f)[..., np.newaxis]], axis=2)  # (NQ,NBE,2)

        aux_rhs_c_3 = -1. / (self.pde.epsilon * self.pde.m) * \
                      (np.einsum('i, ij, ijk, j->jk', self.c_ws, uh_val * vel0_val_c, self.gphi_c[..., 0], self.cellmeasure)
                       + np.einsum('i, ij, ijk, j->jk', self.c_ws, uh_val * vel1_val_c, self.gphi_c[..., 1], self.cellmeasure))  # (NC,cldof)
        aux_rhs_f_3 = 1. / (self.pde.epsilon * self.pde.m) * \
                      np.einsum('i, ijk, jk, ijn, j->jn', self.f_ws, uh_vel_val_f, self.nbdEdge, self.phi_f, self.bdEdgeMeasure)  # (NBE,fldof)

        # # --- assemble the two parts of CH's aux equations
        np.add.at(aux_rv_part0, self.cell2dof, aux_rhs_c_0 + aux_rhs_c_1)
        np.add.at(aux_rv_part1, self.cell2dof, aux_rhs_c_2 + aux_rhs_c_3)
        np.add.at(aux_rv_part0, self.face2dof[self.NeuEdgeIdx_CH, :], aux_rhs_f_1)
        np.add.at(aux_rv_part1, self.face2dof[self.NeuEdgeIdx_CH, :], aux_rhs_f_2)
        np.add.at(aux_rv_part1, self.face2dof[self.bdIndx, :], aux_rhs_f_3)

        # # update the solution of auxiliary equation
        wh_part0 = spsolve(self.StiffMatrix + (self.alpha + self.s / self.pde.epsilon) * self.MassMatrix, aux_rv_part0)
        wh_part1 = spsolve(self.StiffMatrix + (self.alpha + self.s / self.pde.epsilon) * self.MassMatrix, aux_rv_part1)

        # # update the two parts of original CH solution uh
        # --- part0
        orig_rv_part0 = np.zeros((self.dof.number_of_global_dofs(),), dtype=self.ftype)  # (Ndof,)
        wh_val_part0 = self.space.value(wh_part0, self.c_bcs)  # (NQ,NC)
        orig_rhs_c_part0 = - np.einsum('i, ij, ijk, j->jk', self.c_ws, wh_val_part0, self.phi_c, self.cellmeasure)  # (NC,cldof)
        np.add.at(orig_rv_part0, self.cell2dof, orig_rhs_c_part0)
        uh_part0 = spsolve(self.StiffMatrix - self.alpha * self.MassMatrix, orig_rv_part0)

        # --- TODO: The following is the Neumann of <\nabla uh\cdot n, \phi>, but I DON'T know to add which term?
        # TODO: orig_rhs_f = np.einsum('i, ij, ijn, j->jn', self.f_ws, Neumann, self.phi_f, self.NeuEdgeMeasure_CH)
        # TODO: np.add.at(orig_rv, self.face2dof[self.NeuEdgeIdx_CH, :], orig_rhs_f)

        # --- part1
        orig_rv_part1 = np.zeros((self.dof.number_of_global_dofs(),), dtype=self.ftype)  # (Ndof,)
        wh_val_part1 = self.space.value(wh_part1, self.c_bcs)  # (NQ,NC)
        orig_rhs_c_part1 = - np.einsum('i, ij, ijk, j->jk', self.c_ws, wh_val_part1, self.phi_c, self.cellmeasure)  # (NC,cldof)
        np.add.at(orig_rv_part1, self.cell2dof, orig_rhs_c_part1)
        uh_part1 = spsolve(self.StiffMatrix - self.alpha * self.MassMatrix, orig_rv_part1)

        return uh_part0, uh_part1, wh_part0, wh_part1
