#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: FEM_CH_NS_Model2d.py
# @Author: Yongchao Zhang, Northwest University, Xi'an
# @E-mail: yoczhang@nwu.edu.cn
# @Site:
# @Time: Aug 10, 2021
# ---

__doc__ = """
The FEM for coupled Cahn-Hilliard-Navier-Stokes model in 2D. 
"""

import numpy as np
from scipy.sparse import csr_matrix, spdiags, eye, bmat
from fealpy.quadrature import FEMeshIntegralAlg
from scipy.sparse.linalg import spsolve
from fealpy.boundarycondition import DirichletBC
from fealpy.decorator import timer
from fealpy.functionspace import LagrangeFiniteElementSpace
from sym_diff_basis import compute_basis


class FEM_CH_NS_Model2d:
    def __init__(self, pde, mesh, p, dt):
        self.pde = pde
        self.p = p
        self.mesh = mesh
        self.cb = compute_basis(p, mesh)
        self.timemesh, self.dt = self.pde.time_mesh(dt)
        self.itype = self.mesh.itype
        self.ftype = self.mesh.ftype
        self.pde = pde
        self.space = LagrangeFiniteElementSpace(mesh, p)
        self.vspace = LagrangeFiniteElementSpace(mesh, p + 1)  # NS 方程的速度空间
        self.dof = self.space.dof
        self.vdof = self.vspace.dof
        self.cellmeasure = mesh.entity_measure('cell')
        self.integralalg = FEMeshIntegralAlg(self.mesh, p + 4, cellmeasure=self.cellmeasure)
        self.uh = self.space.function()  # CH 方程的解
        self.wh = self.space.function()  # CH 方程中的中间变量
        self.vel0 = self.vspace.function()  # NS 方程中的速度-0分量
        self.vel1 = self.vspace.function()  # NS 方程中的速度-1分量
        self.ph = self.space.function()  # NS 方程中的压力
        self.StiffMatrix = self.space.stiff_matrix()
        self.MassMatrix = self.space.mass_matrix()
        self.vel_SM = self.vspace.stiff_matrix()
        self.vel_MM = self.vspace.mass_matrix()
        self.dt_min = pde.dt_min if hasattr(pde, 'dt_min') else self.dt
        self.s, self.alpha = self.set_CH_Coeff(dt_minimum=self.dt_min)
        self.face2dof = self.dof.face_to_dof()  # (NE,fldof)
        self.cell2dof = self.dof.cell_to_dof()  # (NC,cldof)
        self.vface2dof = self.vdof.face_to_dof()  # (NE,vfldof)
        self.vcell2dof = self.vdof.cell_to_dof()  # (NC,vcldof)
        self.bdIndx = self.mesh.ds.boundary_face_index()  # all the boundary edge index
        self.bdEdgeMeasure = self.mesh.entity_measure('face', index=self.bdIndx)  # (Nneu,)

        # # --- Cahn-Hilliard boundary settings
        self.NeuEdgeIdx_CH = self.set_CH_Neumann_edge()
        self.nNeu_CH = self.mesh.face_unit_normal(index=self.NeuEdgeIdx_CH)  # (NBE,2)
        self.NeuCellIdx_CH = self.mesh.ds.edge2cell[self.NeuEdgeIdx_CH, 0]
        self.NeuLocalIdx_CH = self.mesh.ds.edge2cell[self.NeuEdgeIdx_CH, 2]
        self.NeuEdgeMeasure_CH = self.mesh.entity_measure('face', index=self.NeuEdgeIdx_CH)  # (Nneu,)

        # # --- Navier-Stokes boundary settings
        self.DirEdgeIdx_NS = self.set_NS_Dirichlet_edge()
        self.nDir_NS = self.mesh.face_unit_normal(index=self.DirEdgeIdx_NS)  # (NBE,2)
        self.DirCellIdx_NS = self.mesh.ds.edge2cell[self.DirEdgeIdx_NS, 0]
        self.DirLocalIdx_NS = self.mesh.ds.edge2cell[self.DirEdgeIdx_NS, 2]
        self.DirEdgeMeasure_NS = self.mesh.entity_measure('face', index=self.DirEdgeIdx_NS)  # (Nneu,2)
        self.NeuEdgeIdx_NS = self.bdIndx  # here Neumann edges are all the boundary
        self.nNeu_NS = self.mesh.face_unit_normal(index=self.NeuEdgeIdx_NS)  # (NBE,2)
        self.NeuEdgeMeasure_NS = self.mesh.entity_measure('face', index=self.NeuEdgeIdx_NS)  # (Nneu,2)

        # # --- face-integrator settings
        self.f_bcs, self.f_ws = self.integralalg.faceintegrator.get_quadrature_points_and_weights()  # f_bcs.shape: (NQ,(GD-1)+1)
        self.f_pp_Neu_CH = self.mesh.bc_to_point(self.f_bcs,
                                                 index=self.NeuEdgeIdx_CH)  # f_pp.shape: (NQ,NBE,GD) the physical Gauss points
        self.f_pp_Dir_NS = self.mesh.bc_to_point(self.f_bcs,
                                                 index=self.DirEdgeIdx_NS)  # f_pp.shape: (NQ,NBE,GD) the physical Gauss points
        self.f_pp_Neu_NS = self.mesh.bc_to_point(self.f_bcs,
                                                 index=self.NeuEdgeIdx_NS)  # f_pp.shape: (NQ,NBE,GD) the physical Gauss points

        # # --- cell-integrator settings
        self.c_bcs, self.c_ws = self.integralalg.cellintegrator.get_quadrature_points_and_weights()  # c_bcs.shape: (NQ,GD+1)
        self.c_pp = self.mesh.bc_to_point(self.c_bcs)  # c_pp.shape: (NQ_cell,NC,GD) the physical Gauss points

        # # --- basis-settings
        self.phi_f = self.space.face_basis(self.f_bcs)  # (NQ,1,fldof) 实际上这里可以直接用 pspace.basis(f_bcs), 两个函数的代码是相同的
        self.phi_c = self.space.basis(self.c_bcs)  # (NQ,NC,cldof)
        self.vphi_c = self.vspace.basis(self.c_bcs)  # (NQ,NC,vcldof)
        self.vphi_f = self.vspace.face_basis(self.f_bcs)  # (NQ,NC,vcldof)
        self.gphi_f = self.space.edge_grad_basis(self.f_bcs, self.DirCellIdx_NS, self.DirLocalIdx_NS)  # (NDir,NQ,cldof,GD)
        self.gphi_c = self.space.grad_basis(self.c_bcs)  # (NQ,NC,cldof,GD)

    def set_CH_Coeff(self, dt_minimum=None):
        pde = self.pde
        dt_min = self.dt if dt_minimum is None else dt_minimum
        m = pde.m
        epsilon = pde.epsilon

        if pde.timeScheme in {1, '1', '1st', '1st-order', '1stOrder', 'first', 'first-order', 'firstorder'}:
            s = np.sqrt(4 * epsilon / (m * dt_min))
            alpha = 1. / (2 * epsilon) * (-s + np.sqrt(abs(s ** 2 - 4 * epsilon / (m * self.dt))))
        else:  # time Second-Order
            s = np.sqrt(4 * (3 / 2) * epsilon / (m * dt_min))
            alpha = 1. / (2 * epsilon) * (-s + np.sqrt(abs(s ** 2 - 4 * (3 / 2) * epsilon / (m * self.dt))))
        return s, alpha

    def uh_grad_value_at_faces(self, vh, f_bcs, cellidx, localidx, space=None):
        sp = self.space if space is None else space
        cell2dof = sp.dof.cell2dof
        f_gphi = sp.edge_grad_basis(f_bcs, cellidx, localidx)  # (NE,NQ,cldof,GD)
        val = np.einsum('ik, ijkm->jim', vh[cell2dof[cellidx]], f_gphi)  # (NQ,NE,GD)
        return val

    def grad_free_energy_at_faces(self, uh, f_bcs, idxBdEdge, cellidx, localidx):
        """
        1. Compute the grad of free energy at FACE Gauss-integration points (barycentric coordinates).
        2. In this function, the free energy has NO coefficients.
        -------
        :param uh:
        :param f_bcs: f_bcs.shape: (NQ,(GD-1)+1)
        :return:
        """

        uh_val = self.space.value(uh, f_bcs)[..., idxBdEdge]  # (NQ,NBE)
        guh_val = self.uh_grad_value_at_faces(uh, f_bcs, cellidx, localidx)  # (NQ,NBE,GD)

        guh_val[..., 0] = 3 * uh_val ** 2 * guh_val[..., 0] - guh_val[..., 0]
        guh_val[..., 1] = 3 * uh_val ** 2 * guh_val[..., 1] - guh_val[..., 1]
        return guh_val  # (NQ,NBE,2)

    def grad_free_energy_at_cells(self, uh, c_bcs):
        """
        1. Compute the grad of free energy at CELL Gauss-integration points (barycentric coordinates).
        2. In this function, the free energy has NO coefficients.
        -------
        :param uh:
        :param c_bcs: c_bcs.shape: (NQ,GD+1)
        :return:
        """

        uh_val = self.space.value(uh, c_bcs)  # (NQ,NC)
        guh_val = self.space.grad_value(uh, c_bcs)  # (NQ,NC,2)

        guh_val[..., 0] = 3 * uh_val ** 2 * guh_val[..., 0] - guh_val[..., 0]
        guh_val[..., 1] = 3 * uh_val ** 2 * guh_val[..., 1] - guh_val[..., 1]
        return guh_val  # (NQ,NC,2)

    def CH_NS_Solver_T1stOrder(self):
        pde = self.pde
        timemesh = self.timemesh
        NT = len(timemesh)
        dt = self.dt
        uh = self.uh
        wh = self.wh
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
            self.decoupled_CH_Solver_T1stOrder(uh, wh, vel0, vel1, next_t)
            # print('        -----------------------------------------------')
            # print('        |___ decoupled Navier-Stokes Solver(Time-1st-order): ')
            self.decoupled_NS_Solver_T1stOrder(vel0, vel1, ph, uh_currt, next_t)
            # print('    end of one-looping')

            if nt % max([int(NT / 5), 1]) == 0:
                print('    currt_t = %.4e' % currt_t)
                uh_l2err, uh_h1err, vel_l2err, vel_h1err, ph_l2err = self.currt_error(uh, vel0, vel1, ph, timemesh[nt])
                if np.isnan(uh_l2err) | np.isnan(uh_h1err) | np.isnan(vel_l2err) | np.isnan(vel_h1err) | np.isnan(ph_l2err):
                    print('Some error is nan: breaking the program')
                    break
        print('    # ------------ end the time-looping ------------ #\n')

        # # --- errors
        uh_l2err, uh_h1err, vel_l2err, vel_h1err, ph_l2err = self.currt_error(uh, vel0, vel1, ph, timemesh[-1])
        print('    # ------------ the last errors ------------ #')
        print('    uh_l2err = %.4e, uh_h1err = %.4e' % (uh_l2err, uh_h1err))
        print('    vel_l2err = %.4e, vel_h1err = %.4e, ph_l2err = %.4e' % (vel_l2err, vel_h1err, ph_l2err))

        return uh_l2err, uh_h1err, vel_l2err, vel_h1err, ph_l2err

    def decoupled_CH_Solver_T1stOrder(self, uh, wh, vel0, vel1, next_t):
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
        aux_rv = np.zeros((self.dof.number_of_global_dofs(),), dtype=self.ftype)  # (Ndof,)

        # # aux_rhs_c_0:  -1. / (epsilon * m * dt) * (uh^n,phi)_\Omega
        aux_rhs_c_0 = -1. / (self.pde.epsilon * self.pde.m) * \
                      (1 / self.dt * np.einsum('i, ij, ijk, j->jk', self.c_ws, uh_val, self.phi_c, self.cellmeasure)
                       + np.einsum('i, ij, ijk, j->jk', self.c_ws, f_val_CH, self.phi_c, self.cellmeasure))  # (NC,cldof)

        # # aux_rhs_c_1: -s / epsilon * (\nabla uh^n, \nabla phi)_\Omega
        aux_rhs_c_1 = -self.s / self.pde.epsilon * (
                np.einsum('i, ij, ijk, j->jk', self.c_ws, guh_val_c[..., 0], self.gphi_c[..., 0], self.cellmeasure)
                + np.einsum('i, ij, ijk, j->jk', self.c_ws, guh_val_c[..., 1], self.gphi_c[..., 1], self.cellmeasure))  # (NC,cldof)

        # # aux_rhs_c_2: 1 / epsilon * (\nabla h(uh^n), \nabla phi)_\Omega
        aux_rhs_c_2 = 1. / self.pde.epsilon * (
                np.einsum('i, ij, ijk, j->jk', self.c_ws, grad_free_energy_c[..., 0], self.gphi_c[..., 0], self.cellmeasure)
                + np.einsum('i, ij, ijk, j->jk', self.c_ws, grad_free_energy_c[..., 1], self.gphi_c[..., 1], self.cellmeasure))  # (NC,cldof)

        # # aux_rhs_f_0: (\nabla wh^{n+1}\cdot n, phi)_\Gamma, wh is the solution of auxiliary equation
        aux_rhs_f_0 = self.alpha * np.einsum('i, ij, ijn, j->jn', self.f_ws, Neumann, self.phi_f, self.NeuEdgeMeasure_CH)\
                      + np.einsum('i, ij, ijn, j->jn', self.f_ws, LaplaceNeumann, self.phi_f, self.NeuEdgeMeasure_CH)  # (Nneu,fldof)

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
        uh_vel_val_f = np.concatenate([(uh_val_f*vel0_val_f)[..., np.newaxis],
                                       (uh_val_f*vel1_val_f)[..., np.newaxis]], axis=2)

        aux_rhs_c_3 = -1. / (self.pde.epsilon * self.pde.m) * \
                      (np.einsum('i, ij, ijk, j->jk', self.c_ws, uh_val * vel0_val_c, self.gphi_c[..., 0], self.cellmeasure)
                       + np.einsum('i, ij, ijk, j->jk', self.c_ws, uh_val * vel1_val_c, self.gphi_c[..., 1], self.cellmeasure))  # (NC,cldof)
        aux_rhs_f_3 = 1. / (self.pde.epsilon * self.pde.m) * \
                      np.einsum('i, ijk, jk, ijn, j->jn', self.f_ws, uh_vel_val_f, self.nNeu_CH, self.phi_f, self.bdEdgeMeasure)  # (Nneu,fldof)

        # # --- assemble the CH's aux equation
        np.add.at(aux_rv, self.cell2dof, aux_rhs_c_0 + aux_rhs_c_1 + aux_rhs_c_2 + aux_rhs_c_3)
        np.add.at(aux_rv, self.face2dof[self.NeuEdgeIdx_CH, :], aux_rhs_f_0 + aux_rhs_f_1 + aux_rhs_f_2)
        np.add.at(aux_rv, self.face2dof[self.bdIndx, :], aux_rhs_f_3)

        # # update the solution of auxiliary equation
        wh[:] = spsolve(self.StiffMatrix + (self.alpha + self.s / self.pde.epsilon) * self.MassMatrix, aux_rv)

        # # update the original CH solution uh, we do not need to change the code
        orig_rv = np.zeros((self.dof.number_of_global_dofs(),), dtype=self.ftype)  # (Ndof,)
        wh_val = self.space.value(wh, self.c_bcs)  # (NQ,NC)
        orig_rhs_c = - np.einsum('i, ij, ijk, j->jk', self.c_ws, wh_val, self.phi_c, self.cellmeasure)  # (NC,cldof)
        orig_rhs_f = np.einsum('i, ij, ijn, j->jn', self.f_ws, Neumann, self.phi_f, self.NeuEdgeMeasure_CH)
        np.add.at(orig_rv, self.cell2dof, orig_rhs_c)
        np.add.at(orig_rv, self.face2dof[self.NeuEdgeIdx_CH, :], orig_rhs_f)
        uh[:] = spsolve(self.StiffMatrix - self.alpha * self.MassMatrix, orig_rv)

    def decoupled_CH_Solver_T2ndOrder(self, uh, uh_star, vel_star_0, vel_star_1, next_t):
        """
        The decoupled-Cahn-Hilliard-solver for the all system.
        :param uh: The value of the solution 'phi' of Cahn-Hilliard equation: stored the n-th(time) value, and to update the (n+1)-th value.
        :param uh_star: 2*uh - last_uh
        :param vel_star_0: 2*vel0 - last_vel0, where vel0 is the fist-component of NS's velocity (the n-th(time) value),
                           last_vel0 is the (n-1)-th(time) value.
        :param vel_star_1: 2*vel1 - last_vel1, where vel0 is the second-component of NS's velocity (the n-th(time) value),
                           last_vel1 is the (n-1)-th(time) value.
        :param next_t: Next time.
        :return: Updated uh, wh.
        """

        grad_free_energy_c = self.pde.epsilon / self.pde.eta ** 2 * self.grad_free_energy_at_cells(uh_star, self.c_bcs)  # (NQ,NC,2)
        grad_free_energy_f = self.pde.epsilon / self.pde.eta ** 2 * \
                             self.grad_free_energy_at_faces(uh_star, self.f_bcs, self.NeuEdgeIdx_CH, self.NeuCellIdx_CH,
                                                            self.NeuLocalIdx_CH)  # (NQ,NE,2)
        last_uh = 2 * uh - uh_star
        uh_hat = 2 * uh - 0.5 * last_uh
        uh_hat_val = self.space.value(uh_hat, self.c_bcs)  # (NQ,NC)
        uh_star_val = self.space.value(uh_star, self.c_bcs)  # (NQ,NC)
        guh_star_val_c = self.space.grad_value(uh_star, self.c_bcs)  # (NQ,NC,2)
        guh_star_val_f = self.uh_grad_value_at_faces(uh_star, self.f_bcs, self.NeuCellIdx_CH, self.NeuLocalIdx_CH)  # (NQ,NNeu,2)

        Neumann = self.pde.neumann_CH(self.f_pp_Neu_CH, next_t, self.nNeu_CH)  # (NQ,NE)
        LaplaceNeumann = self.pde.laplace_neumann_CH(self.f_pp_Neu_CH, next_t, self.nNeu_CH)  # (NQ,NE)
        f_val_CH = self.pde.source_CH(self.c_pp, next_t, self.pde.m, self.pde.epsilon, self.pde.eta)  # (NQ,NC)

        # # get the auxiliary equation Right-hand-side-Vector
        aux_rv = np.zeros((self.dof.number_of_global_dofs(),), dtype=self.ftype)  # (Ndof,)

        # # aux_rhs_c_0:  -1. / (epsilon * m * dt) * (uh_hat_val + g^{n+1} ,phi)_\Omega
        aux_rhs_c_0 = -1. / (self.pde.epsilon * self.pde.m) * \
                      (1 / self.dt * np.einsum('i, ij, ijk, j->jk', self.c_ws, uh_hat_val, self.phi_c, self.cellmeasure)
                       + np.einsum('i, ij, ijk, j->jk', self.c_ws, f_val_CH, self.phi_c, self.cellmeasure))  # (NC,cldof)

        # # aux_rhs_c_1: -s / epsilon * (\nabla uh^n, \nabla phi)_\Omega
        aux_rhs_c_1 = -self.s / self.pde.epsilon * (
                np.einsum('i, ij, ijk, j->jk', self.c_ws, guh_star_val_c[..., 0], self.gphi_c[..., 0], self.cellmeasure)
                + np.einsum('i, ij, ijk, j->jk', self.c_ws, guh_star_val_c[..., 1], self.gphi_c[..., 1],
                            self.cellmeasure))  # (NC,cldof)

        # # aux_rhs_c_2: 1 / epsilon * (\nabla h(uh^n), \nabla phi)_\Omega
        aux_rhs_c_2 = 1. / self.pde.epsilon * (
                np.einsum('i, ij, ijk, j->jk', self.c_ws, grad_free_energy_c[..., 0], self.gphi_c[..., 0], self.cellmeasure)
                + np.einsum('i, ij, ijk, j->jk', self.c_ws, grad_free_energy_c[..., 1], self.gphi_c[..., 1],
                            self.cellmeasure))  # (NC,cldof)

        # # aux_rhs_f_0: (\nabla wh^{n+1}\cdot n, phi)_\Gamma, wh is the solution of auxiliary equation
        aux_rhs_f_0 = self.alpha * np.einsum('i, ij, ijn, j->jn', self.f_ws, Neumann, self.phi_f, self.NeuEdgeMeasure_CH) \
                      + np.einsum('i, ij, ijn, j->jn', self.f_ws, LaplaceNeumann, self.phi_f,
                                  self.NeuEdgeMeasure_CH)  # (Nneu,fldof)

        # # aux_rhs_f_1: s / epsilon * (\nabla uh^n \cdot n, phi)_\Gamma
        aux_rhs_f_1 = self.s / self.pde.epsilon * np.einsum('i, ijk, jk, ijn, j->jn', self.f_ws, guh_star_val_f, self.nNeu_CH,
                                                            self.phi_f, self.NeuEdgeMeasure_CH)  # (Nneu,fldof)

        # # aux_rhs_f_2: -1 / epsilon * (\nabla h(uh^n) \cdot n, phi)_\Gamma
        aux_rhs_f_2 = -1. / self.pde.epsilon * np.einsum('i, ijk, jk, ijn, j->jn', self.f_ws, grad_free_energy_f, self.nNeu_CH,
                                                         self.phi_f, self.NeuEdgeMeasure_CH)  # (Nneu,fldof)

        # # --- now, we add the NS term
        vel0_star_val_c = self.vspace.value(vel_star_0, self.c_bcs)  # (NQ,NC)
        vel1_star_val_c = self.vspace.value(vel_star_1, self.c_bcs)  # (NQ,NC)
        vel0_star_val_f = self.vspace.value(vel_star_0, self.f_bcs)[..., self.bdIndx]  # (NQ,NBE)
        vel1_star_val_f = self.vspace.value(vel_star_1, self.f_bcs)[..., self.bdIndx]  # (NQ,NBE)

        uh_star_val_f = self.space.value(uh_star, self.f_bcs)[..., self.bdIndx]  # (NQ,NBE)
        uh_vel_val_f = np.concatenate([(uh_star_val_f * vel0_star_val_f)[..., np.newaxis],
                                       (uh_star_val_f * vel1_star_val_f)[..., np.newaxis]], axis=2)

        aux_rhs_c_3 = -1. / (self.pde.epsilon * self.pde.m) * \
                      (np.einsum('i, ij, ijk, j->jk', self.c_ws, uh_star_val * vel0_star_val_c, self.gphi_c[..., 0], self.cellmeasure)
                       + np.einsum('i, ij, ijk, j->jk', self.c_ws, uh_star_val * vel1_star_val_c, self.gphi_c[..., 1],
                                   self.cellmeasure))  # (NC,cldof)
        aux_rhs_f_3 = 1. / (self.pde.epsilon * self.pde.m) * \
                      np.einsum('i, ijk, jk, ijn, j->jn', self.f_ws, uh_vel_val_f, self.nNeu_CH, self.phi_f,
                                self.bdEdgeMeasure)  # (Nneu,fldof)

        # # --- assemble the CH's aux equation
        np.add.at(aux_rv, self.cell2dof, aux_rhs_c_0 + aux_rhs_c_1 + aux_rhs_c_2 + aux_rhs_c_3)
        np.add.at(aux_rv, self.face2dof[self.NeuEdgeIdx_CH, :], aux_rhs_f_0 + aux_rhs_f_1 + aux_rhs_f_2)
        np.add.at(aux_rv, self.face2dof[self.bdIndx, :], aux_rhs_f_3)

        # # update the solution of auxiliary equation
        wh = self.space.function()
        wh[:] = spsolve(self.StiffMatrix + (self.alpha + self.s / self.pde.epsilon) * self.MassMatrix, aux_rv)

        # # update the original CH solution uh, we do not need to change the code
        orig_rv = np.zeros((self.dof.number_of_global_dofs(),), dtype=self.ftype)  # (Ndof,)
        wh_val = self.space.value(wh, self.c_bcs)  # (NQ,NC)
        orig_rhs_c = - np.einsum('i, ij, ijk, j->jk', self.c_ws, wh_val, self.phi_c, self.cellmeasure)  # (NC,cldof)
        orig_rhs_f = np.einsum('i, ij, ijn, j->jn', self.f_ws, Neumann, self.phi_f, self.NeuEdgeMeasure_CH)
        np.add.at(orig_rv, self.cell2dof, orig_rhs_c)
        np.add.at(orig_rv, self.face2dof[self.NeuEdgeIdx_CH, :], orig_rhs_f)
        uh[:] = spsolve(self.StiffMatrix - self.alpha * self.MassMatrix, orig_rv)

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

        grad_vel0_f = self.uh_grad_value_at_faces(vel0, self.f_bcs, self.DirCellIdx_NS, self.DirLocalIdx_NS, space=self.vspace)  # grad_vel0: (NQ,NDir,GD)
        grad_vel1_f = self.uh_grad_value_at_faces(vel1, self.f_bcs, self.DirCellIdx_NS, self.DirLocalIdx_NS, space=self.vspace)  # grad_vel1: (NQ,NDir,GD)

        # for cell-integration
        vel0_val = self.vspace.value(vel0, self.c_bcs)  # (NQ,NC)
        vel1_val = self.vspace.value(vel1, self.c_bcs)

        nolinear_val = self.NSNolinearTerm(vel0, vel1, self.c_bcs)  # last_nolinear_val.shape: (NQ,NC,GD)
        nolinear_val0 = nolinear_val[..., 0]  # (NQ,NC)
        nolinear_val1 = nolinear_val[..., 1]  # (NQ,NC)

        velDir_val = self.pde.dirichlet_NS(self.f_pp_Dir_NS, next_t)  # (NQ,NDir,GD)
        f_val_NS = self.pde.source_NS(self.c_pp, next_t, self.pde.nu, self.pde.epsilon, self.pde.eta)  # (NQ,NC,GD)
        Neumann_0 = self.pde.neumann_0_NS(self.f_pp_Neu_NS, next_t, self.nNeu_NS)  # (NQ,NE)
        Neumann_1 = self.pde.neumann_1_NS(self.f_pp_Neu_NS, next_t, self.nNeu_NS)  # (NQ,NE)

        # # --- to update the pressure value --- # #
        # # to get the Pressure's Right-hand Vector
        prv = np.zeros((self.dof.number_of_global_dofs(),), dtype=self.ftype)  # (Npdof,)

        # for Dirichlet faces integration
        dir_int0 = -1 / self.dt * np.einsum('i, ijk, jk, ijn, j->jn', self.f_ws, velDir_val, self.nDir_NS, self.phi_f, self.DirEdgeMeasure_NS)  # (NDir,fldof)
        dir_int1 = - self.pde.nu * (np.einsum('i, j, ij, jin, j->jn', self.f_ws, self.nDir_NS[:, 1], grad_vel1_f[..., 0] - grad_vel0_f[..., 1],
                                              self.gphi_f[..., 0], self.DirEdgeMeasure_NS)
                                    + np.einsum('i, j, ij, jin, j->jn', self.f_ws, -self.nDir_NS[:, 0], grad_vel1_f[..., 0] - grad_vel0_f[..., 1],
                                                self.gphi_f[..., 1], self.DirEdgeMeasure_NS))  # (NDir,cldof)
        # for cell integration
        cell_int0 = 1 / self.dt * (np.einsum('i, ij, ijk, j->jk', self.c_ws, vel0_val, self.gphi_c[..., 0], self.cellmeasure)
                                   + np.einsum('i, ij, ijk, j->jk', self.c_ws, vel1_val, self.gphi_c[..., 1], self.cellmeasure))  # (NC,cldof)
        cell_int1 = -(np.einsum('i, ij, ijk, j->jk', self.c_ws, nolinear_val0, self.gphi_c[..., 0], self.cellmeasure)
                      + np.einsum('i, ij, ijk, j->jk', self.c_ws, nolinear_val1, self.gphi_c[..., 1], self.cellmeasure))  # (NC,cldof)
        cell_int2 = (np.einsum('i, ij, ijk, j->jk', self.c_ws, f_val_NS[..., 0], self.gphi_c[..., 0], self.cellmeasure)
                     + np.einsum('i, ij, ijk, j->jk', self.c_ws, f_val_NS[..., 1], self.gphi_c[..., 1], self.cellmeasure))  # (NC,cldof)

        # # --- now, we add the CH term
        uh_val = self.space.value(uh, self.c_bcs)  # (NQ,NC)
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

        cell_int3 = - (np.einsum('i, ij, ijk, j->jk', self.c_ws, CH_term_val0, self.gphi_c[..., 0], self.cellmeasure)
                       + np.einsum('i, ij, ijk, j->jk', self.c_ws, CH_term_val1, self.gphi_c[..., 1], self.cellmeasure))  # (NC,ldof)

        # # --- 1. assemble the NS's pressure equation
        np.add.at(prv, self.face2dof[self.DirEdgeIdx_NS, :], dir_int0)
        np.add.at(prv, self.cell2dof[self.DirCellIdx_NS, :], dir_int1)
        np.add.at(prv, self.cell2dof, cell_int0 + cell_int1 + cell_int2 + cell_int3)

        # # --- 2. solve the NS's pressure equation
        plsm = self.StiffMatrix

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
        vlm0 = 1 / self.dt * self.vel_MM + self.pde.nu * self.vel_SM
        vlm1 = vlm0.copy()

        # # to get the u's Right-hand Vector
        def dir_u0(p):
            return self.pde.dirichlet_NS(p, next_t)[..., 0]

        def dir_u1(p):
            return self.pde.dirichlet_NS(p, next_t)[..., 1]

        # # --- assemble the first-component of Velocity-Right-Vector
        vrv0 = np.zeros((self.vdof.number_of_global_dofs(),), dtype=self.ftype)  # (Nvdof,)
        vrv0_c = np.einsum('i, ij, ijk, j->jk', self.c_ws, vel0_val / self.dt - grad_ph[..., 0] - nolinear_val0
                           + f_val_NS[..., 0] - CH_term_val0, self.vphi_c, self.cellmeasure)  # (NC,clodf)
        vrv0_f = self.pde.nu * np.einsum('i, ij, ijn, j->jn', self.f_ws, Neumann_0, self.vphi_f, self.NeuEdgeMeasure_NS)
        np.add.at(vrv0, self.vcell2dof, vrv0_c)
        # np.add.at(vrv0, self.vface2dof[self.NeuEdgeIdx_NS, :], vrv0_f)
        v0_bc = DirichletBC(self.vspace, dir_u0, threshold=self.DirEdgeIdx_NS)
        vlm0, vrv0 = v0_bc.apply(vlm0, vrv0)
        vel0[:] = spsolve(vlm0, vrv0).reshape(-1)

        # # --- assemble the second-component of Velocity-Right-Vector
        vrv1 = np.zeros((self.vdof.number_of_global_dofs(),), dtype=self.ftype)  # (Nvdof,)
        vrv1_c = np.einsum('i, ij, ijk, j->jk', self.c_ws, vel1_val / self.dt - grad_ph[..., 1] - nolinear_val1
                           + f_val_NS[..., 1] - CH_term_val1, self.vphi_c, self.cellmeasure)  # (NC,clodf)
        vrv1_f = self.pde.nu * np.einsum('i, ij, ijn, j->jn', self.f_ws, Neumann_1, self.vphi_f, self.NeuEdgeMeasure_NS)
        np.add.at(vrv1, self.vcell2dof, vrv1_c)
        # np.add.at(vrv1, self.vface2dof[self.NeuEdgeIdx_NS, :], vrv1_f)
        v1_bc = DirichletBC(self.vspace, dir_u1, threshold=self.DirEdgeIdx_NS)
        vlm1, vrv1 = v1_bc.apply(vlm1, vrv1)
        vel1[:] = spsolve(vlm1, vrv1).reshape(-1)

    def NSNolinearTerm(self, uh0, uh1, bcs):
        vspace = self.vspace
        val0 = vspace.value(uh0, bcs)  # val0.shape: (NQ,NC)
        val1 = vspace.value(uh1, bcs)  # val1.shape: (NQ,NC)
        gval0 = vspace.grad_value(uh0, bcs)  # gval0.shape: (NQ,NC,2)
        gval1 = vspace.grad_value(uh1, bcs)

        NSNolinear = np.empty(gval0.shape, dtype=self.ftype)  # NSNolinear.shape: (NQ,NC,2)

        NSNolinear[..., 0] = val0 * gval0[..., 0] + val1 * gval0[..., 1]
        NSNolinear[..., 1] = val0 * gval1[..., 0] + val1 * gval1[..., 1]
        return NSNolinear

    def currt_error(self, uh, vel0, vel1, ph, t):
        pde = self.pde

        # # --- Cahn-Hilliard equation's errors
        def currt_solution_CH(p):
            return pde.solution_CH(p, t)
        l2err_CH = self.space.integralalg.L2_error(currt_solution_CH, uh)

        def currt_grad_solution_CH(p):
            return pde.gradient_CH(p, t)
        h1err_CH = self.space.integralalg.L2_error(currt_grad_solution_CH, uh.grad_value)

        # # --- Navier-Stokes equation's errors
        def currt_pressure_NS(p):
            return pde.pressure_NS(p, t)
        p_l2err_NS = self.space.integralalg.L2_error(currt_pressure_NS, ph)

        def currt_v0_NS(p):
            return pde.velocity_NS(p, t)[..., 0]
        v0_l2err_NS = self.vspace.integralalg.L2_error(currt_v0_NS, vel0)

        def currt_v1_NS(p):
            return pde.velocity_NS(p, t)[..., 1]
        v1_l2err_NS = self.vspace.integralalg.L2_error(currt_v1_NS, vel1)

        def currt_grad_velocity0_NS(p):
            return pde.grad_velocity0_NS(p, t)
        v0_h1err_NS = self.vspace.integralalg.L2_error(currt_grad_velocity0_NS, vel0.grad_value)

        def currt_grad_velocity1_NS(p):
            return pde.grad_velocity1_NS(p, t)
        v1_h1err_NS = self.vspace.integralalg.L2_error(currt_grad_velocity1_NS, vel1.grad_value)

        return l2err_CH, h1err_CH, np.sqrt(v0_l2err_NS**2 + v1_l2err_NS**2), np.sqrt(v0_h1err_NS**2 + v1_h1err_NS**2), p_l2err_NS

    def set_NS_Dirichlet_edge(self, idxDirEdge=None):
        if idxDirEdge is not None:
            return idxDirEdge

        mesh = self.mesh
        edge2cell = mesh.ds.edge_to_cell()
        isBdEdge = (edge2cell[:, 0] == edge2cell[:, 1])  # (NE,), the bool vars, to get the boundary edges

        isDirEdge = isBdEdge  # here, we set all the boundary edges are Dir edges
        idxDirEdge, = np.nonzero(isDirEdge)  # (NE_Dir,)

        return idxDirEdge

    def set_CH_Neumann_edge(self, idxNeuEdge=None):
        if idxNeuEdge is not None:
            return idxNeuEdge

        mesh = self.mesh
        edge2cell = mesh.ds.edge_to_cell()
        bdEdge = (edge2cell[:, 0] == edge2cell[:, 1])  # the bool vars, to get the boundary edges

        isNeuEdge = bdEdge  # here, we first set all the boundary edges are Neu edges
        idxNeuEdge, = np.nonzero(isNeuEdge)  # (NE_Dir,)
        return idxNeuEdge
