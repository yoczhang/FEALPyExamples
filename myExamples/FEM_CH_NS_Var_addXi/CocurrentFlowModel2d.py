#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: CocurrentFlowModel2d.py
# @Author: Yongchao Zhang
# @Institution: Northwest University, Xi'an, Shaanxi, China
# @E-mail: yoczhang@126.com, yoczhang@nwu.edu.cn
# @Site: 
# @Time: Feb 16, 2022
# ---

__doc__ = """
The fealpy-FEM program for Variable-Coefficient coupled Cahn-Hilliard-Navier-Stokes equation, 
add the solver for \\xi.
  |___ Co-current flow problem.
"""


import numpy as np
from scipy.sparse import csr_matrix, spdiags, identity, eye, bmat
# from fealpy.quadrature import FEMeshIntegralAlg
from scipy.sparse.linalg import spsolve
from fealpy.boundarycondition import DirichletBC
# from fealpy.decorator import timer
# from fealpy.functionspace import LagrangeFiniteElementSpace
# from sym_diff_basis import compute_basis
from FEM_CH_NS_Model2d import FEM_CH_NS_Model2d


class CocurrentFlowModel2d(FEM_CH_NS_Model2d):
    """
        注意:
            本程序所参考的数值格式是按照 $D(u)=\nabla u + \nabla u^T$ 来写的,
            如果要调整为 $D(u)=(\nabla u + \nabla u^T)/2$, 需要调整的地方太多了,
            所以为了保证和数值格式的一致性, 本程序也是严格按照 $D(u)=\nabla u + \nabla u^T$ 来编写的.
        """

    def __init__(self, pde, mesh, p, dt):
        super(CocurrentFlowModel2d, self).__init__(pde, mesh, p, dt)
        self.wh_part0 = self.space.function()
        self.wh_part1 = self.space.function()
        self.uh_part0 = self.space.function()
        self.uh_part1 = self.space.function()
        self.uh_last_part0 = self.space.function()
        self.uh_last_part1 = self.space.function()
        self.ph_part0 = self.space.function()
        self.ph_part1 = self.space.function()
        self.vel0_part0 = self.vspace.function()
        self.vel1_part0 = self.vspace.function()
        self.vel0_part1 = self.vspace.function()
        self.vel1_part1 = self.vspace.function()
        self.auxVel0_part0 = self.vspace.function()
        self.auxVel1_part0 = self.vspace.function()
        self.auxVel0_part1 = self.vspace.function()
        self.auxVel1_part1 = self.vspace.function()

        self.grad_mu_val = 0.  # 此项在 `update_mu_and_Xi()` 中更新.
        self.rho_bar_n = 0  # 此项在 `decoupled_NS_addXi_Solver_T1stOrder()` 中更新: 为了获得第 n 时间层的取值 (在 `update_mu_and_Xi()` 会用到).
        self.nu_bar_n = 0  # 此项在 `decoupled_NS_addXi_Solver_T1stOrder()` 中更新: 为了获得第 n 时间层的取值 (在 `update_mu_and_Xi()` 会用到).
        self.R_n = 0.  # 此项在 `update_mu_and_Xi()` 中更新.
        self.C0 = 1.  # 此项在 `update_mu_and_Xi()` 中, 以保证 E_n = \int H(\phi) + C0 > 0.
        self.Xi = 1.  # 此项在 `update_mu_and_Xi()` 中更新.
        self.s, self.alpha = self.set_CH_Coeff(dt_minimum=self.dt_min)

        if hasattr(self, 'idxNotPeriodicEdge') is False:
            self.idxPeriodicEdge0, self.idxPeriodicEdge1, self.idxNotPeriodicEdge = self.set_periodic_edge()

        # |--- setting periodic dofs
        self.periodicDof0, self.periodicDof1, self.notPeriodicDof = self.set_boundaryDofs(self.dof)
        #   |___ the phi_h- and p_h-related variables periodic dofs (using p-order polynomial)
        self.vPeriodicDof0, self.vPeriodicDof1, self.vNotPeriodicDof = self.set_boundaryDofs(self.vdof)
        #   |___ the velocity-related variables periodic dofs (using (p+1)-order polynomial)

        # |--- CH: setting algebraic system for periodic boundary condition
        self.auxM_CH = self.StiffMatrix + (self.alpha + self.s / self.pde.epsilon) * self.MassMatrix  # csr_matrix
        self.auxPeriodicM_CH = None
        self.orgM_CH = self.StiffMatrix - self.alpha * self.MassMatrix  # csr_matrix
        self.orgPeriodicM_CH = None

        # |--- NS: setting algebraic system for periodic boundary condition
        self.plsm = 1. / min(self.pde.rho0, self.pde.rho1) * self.StiffMatrix
        self.pPeriodicM_NS = None

        self.auxVLM = 1. / self.dt * self.vel_MM
        self.vAuxPeriodicM_NS = None

        self.VLM = 1. / self.dt * self.vel_MM + max(self.pde.nu0 / self.pde.rho0, self.pde.nu1 / self.pde.rho1) * self.vel_SM
        self.vOrgPeriodicM_NS = None

    def CH_NS_addXi_Solver_T1stOrder(self):
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
        print('    s = %.4e,  alpha = %.4e,  m = %.4e,  epsilon = %.4e,  eta = %.4e'
              % (self.s, self.alpha, self.pde.m, self.pde.epsilon, self.pde.eta))
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
        self.uh_part0[:] = uh[:]
        #    |___ 这里只赋值 uh_part0, 首先在下面的时间循环中赋值给 self.uh_last_part0,
        #         |___ 接着是为了 `第一次` 计算 self.rho_bar_n 与 self.nu_bar_n 时直接取到 uh 的初始值.
        uh_last = uh.copy()
        for nt in range(NT - 1):
            currt_t = timemesh[nt]
            next_t = currt_t + dt

            # # --- decoupled solvers, updated the discrete-solutions to the next-time
            self.uh_last_part0[:] = self.uh_part0[:]
            self.uh_last_part1[:] = self.uh_part1[:]
            # print('        |___ decoupled Cahn-Hilliard Solver(Time-1st-order): ')
            self.decoupled_CH_addXi_Solver_T1stOrder(uh, wh, vel0, vel1, next_t)
            # print('        -----------------------------------------------')
            # print('        |___ decoupled Navier-Stokes Solver(Time-1st-order): ')
            self.decoupled_NS_addXi_Solver_T1stOrder(vel0, vel1, ph, uh, uh_last, next_t)
            Xi = self.update_mu_and_Xi(uh, next_t)

            # |--- update the values
            uh_last[:] = uh[:]
            uh[:] = self.uh_part0[:] + Xi * self.uh_part1[:]
            wh[:] = self.wh_part0[:] + Xi * self.wh_part1[:]
            ph[:] = self.ph_part0[:] + Xi * self.ph_part1[:]
            vel0[:] = self.vel0_part0[:] + Xi * self.vel0_part1[:]
            vel1[:] = self.vel1_part0[:] + Xi * self.vel1_part1[:]
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
        :return: Updated uh_part*, wh_part*.
        """

        pde = self.pde
        wh_part0 = self.wh_part0
        wh_part1 = self.wh_part1
        uh_part0 = self.uh_part0
        uh_part1 = self.uh_part1

        # |--- uh using the true solution
        def solution_CH(p):
            return pde.solution_CH(p, next_t)
        uh[:] = self.space.interpolation(solution_CH)

        # |--- grad-free-energy: \nabla h(phi) = epsilon/eta^2*3*phi^2*(\nabla phi) - (\nabla phi)
        #     |___ where h(phi) = epsilon/eta^2*phi*(phi^2-1)
        grad_free_energy_c = pde.epsilon / pde.eta ** 2 * self.grad_free_energy_at_cells(uh, self.c_bcs)  # (NQ,NC,2)
        grad_free_energy_f = (pde.epsilon / pde.eta ** 2
                              * self.grad_free_energy_at_faces(uh, self.f_bcs, self.NeuEdgeIdx_CH, self.NeuCellIdx_CH,
                                                               self.NeuLocalIdx_CH))  # (NQ,NE,2)
        uh_val = self.space.value(uh, self.c_bcs)  # (NQ,NC)
        guh_val_c = self.space.grad_value(uh, self.c_bcs)  # (NQ,NC,2)
        guh_val_f = self.uh_grad_value_at_faces(uh, self.f_bcs, self.NeuCellIdx_CH, self.NeuLocalIdx_CH)  # (NQ,NNeu,2)

        Neumann = pde.neumann_CH(self.f_pp_Neu_CH, next_t, self.nNeu_CH)  # (NQ,NE)
        LaplaceNeumann = pde.laplace_neumann_CH(self.f_pp_Neu_CH, next_t, self.nNeu_CH)  # (NQ,NE)
        f_val_CH = pde.source_CH(self.c_pp, next_t, pde.m, pde.epsilon, pde.eta)  # (NQ,NC)

        # |--- get the auxiliary equation Right-hand-side-Vector
        aux_rv_part0 = np.zeros((self.dof.number_of_global_dofs(),), dtype=self.ftype)  # (Ndof,)
        aux_rv_part1 = np.zeros((self.dof.number_of_global_dofs(),), dtype=self.ftype)  # (Ndof,)

        # |--- aux_rhs_c_0:  -1. / (epsilon * m) * (uh^n/dt + g^{n+1},phi)_\Omega
        aux_rhs_c_0 = -1. / (pde.epsilon * pde.m) * np.einsum('i, ij, ijk, j->jk', self.c_ws, 1 / self.dt * uh_val
                                                              + f_val_CH, self.phi_c, self.cellmeasure)  # (NC,cldof)

        # |--- aux_rhs_c_1: -s / epsilon * (\nabla uh^n, \nabla phi)_\Omega
        aux_rhs_c_1 = -self.s / pde.epsilon * (
                np.einsum('i, ijm, ijkm, j->jk', self.c_ws, guh_val_c, self.gphi_c, self.cellmeasure))  # (NC,cldof)

        # |--- aux_rhs_c_2: 1 / epsilon * (\nabla h(uh^n), \nabla phi)_\Omega
        aux_rhs_c_2 = 1. / pde.epsilon * (np.einsum('i, ijm, ijkm, j->jk', self.c_ws, grad_free_energy_c,
                                                    self.gphi_c, self.cellmeasure))  # (NC,cldof)

        # |--- aux_rhs_f_0: (\nabla wh^{n+1}\cdot n, phi)_\Gamma, wh is the solution of auxiliary equation
        aux_rhs_f_0 = np.einsum('i, ij, ijn, j->jn', self.f_ws, self.alpha * Neumann + LaplaceNeumann, self.phi_f, self.NeuEdgeMeasure_CH)  # (Nneu,fldof)
        #          |___ This term will add to aux_rv_part0

        # |--- aux_rhs_f_1: s / epsilon * (\nabla uh^n \cdot n, phi)_\Gamma
        aux_rhs_f_1 = self.s / pde.epsilon * np.einsum('i, ijk, jk, ijn, j->jn', self.f_ws, guh_val_f, self.nNeu_CH,
                                                       self.phi_f, self.NeuEdgeMeasure_CH)  # (Nneu,fldof)

        # |--- aux_rhs_f_2: -1 / epsilon * (\nabla h(uh^n) \cdot n, phi)_\Gamma
        aux_rhs_f_2 = -1. / pde.epsilon * np.einsum('i, ijk, jk, ijn, j->jn', self.f_ws, grad_free_energy_f, self.nNeu_CH,
                                                    self.phi_f, self.NeuEdgeMeasure_CH)  # (Nneu,fldof)

        # |--- now, we add the NS term
        vel0_val_c = self.vspace.value(vel0, self.c_bcs)  # (NQ,NC)
        vel1_val_c = self.vspace.value(vel1, self.c_bcs)  # (NQ,NC)
        vel0_val_f = self.vspace.value(vel0, self.f_bcs)[..., self.NeuEdgeIdx_CH]  # (NQ,Nneu)
        vel1_val_f = self.vspace.value(vel1, self.f_bcs)[..., self.NeuEdgeIdx_CH]  # (NQ,Nneu)

        uh_val_f = self.space.value(uh, self.f_bcs)[..., self.NeuEdgeIdx_CH]  # (NQ,Nneu)
        uh_vel_val_f = np.concatenate([(uh_val_f * vel0_val_f)[..., np.newaxis],
                                       (uh_val_f * vel1_val_f)[..., np.newaxis]], axis=2)  # (NQ,Nneu,2)

        aux_rhs_c_3 = (-1. / (pde.epsilon * pde.m) *
                       (np.einsum('i, ij, ijk, j->jk', self.c_ws, uh_val * vel0_val_c, self.gphi_c[..., 0], self.cellmeasure) +
                        np.einsum('i, ij, ijk, j->jk', self.c_ws, uh_val * vel1_val_c, self.gphi_c[..., 1], self.cellmeasure)))  # (NC,cldof)
        aux_rhs_f_3 = (1. / (pde.epsilon * pde.m) *
                       np.einsum('i, ijk, jk, ijn, j->jn', self.f_ws, uh_vel_val_f, self.nNeu_CH, self.phi_f, self.NeuEdgeMeasure_CH))  # (NBE,fldof)

        # |--- assemble the two parts of CH's aux equations
        np.add.at(aux_rv_part0, self.cell2dof, aux_rhs_c_0 + aux_rhs_c_1)
        np.add.at(aux_rv_part0, self.face2dof[self.NeuEdgeIdx_CH, :], aux_rhs_f_0 + aux_rhs_f_1)
        np.add.at(aux_rv_part1, self.cell2dof, aux_rhs_c_2 + aux_rhs_c_3)
        np.add.at(aux_rv_part1, self.face2dof[self.NeuEdgeIdx_CH, :], aux_rhs_f_2 + aux_rhs_f_3)

        if self.auxPeriodicM_CH is None:
            aux_rv_part0, aux_rv_part1, auxPeriodicM = self.set_periodicAlgebraicSystem(self.dof, aux_rv_part0, aux_rv_part1, self.auxM_CH)
            self.auxPeriodicM_CH = auxPeriodicM.copy()
        else:
            auxPeriodicM = self.auxPeriodicM_CH
            aux_rv_part0, aux_rv_part1, _ = self.set_periodicAlgebraicSystem(self.dof, aux_rv_part0, aux_rv_part1)

        wh_part0[:] = spsolve(auxPeriodicM, aux_rv_part0)
        wh_part1[:] = spsolve(auxPeriodicM, aux_rv_part1)

        # |--- update the two parts of original CH solution uh
        #      |___ part0
        orig_rv_part0 = np.zeros((self.dof.number_of_global_dofs(),), dtype=self.ftype)  # (Ndof,)
        wh_val_part0 = self.space.value(wh_part0, self.c_bcs)  # (NQ,NC)
        orig_rhs_c_part0 = - np.einsum('i, ij, ijk, j->jk', self.c_ws, wh_val_part0, self.phi_c, self.cellmeasure)  # (NC,cldof)
        orig_rhs_f_part0 = np.einsum('i, ij, ijn, j->jn', self.f_ws, Neumann, self.phi_f, self.NeuEdgeMeasure_CH)  # (Nneu,fldof)
        np.add.at(orig_rv_part0, self.cell2dof, orig_rhs_c_part0)
        np.add.at(orig_rv_part0, self.face2dof[self.NeuEdgeIdx_CH, :], orig_rhs_f_part0)

        #      |___ part1
        orig_rv_part1 = np.zeros((self.dof.number_of_global_dofs(),), dtype=self.ftype)  # (Ndof,)
        wh_val_part1 = self.space.value(wh_part1, self.c_bcs)  # (NQ,NC)
        orig_rhs_c_part1 = - np.einsum('i, ij, ijk, j->jk', self.c_ws, wh_val_part1, self.phi_c, self.cellmeasure)  # (NC,cldof)
        np.add.at(orig_rv_part1, self.cell2dof, orig_rhs_c_part1)

        if self.orgPeriodicM_CH is None:
            orig_rv_part0, orig_rv_part1, orgPeriodicM = self.set_periodicAlgebraicSystem(self.dof, orig_rv_part0, orig_rv_part1, self.orgM_CH)
            self.orgPeriodicM_CH = orgPeriodicM.copy()
        else:
            orgPeriodicM = self.orgPeriodicM_CH
            orig_rv_part0, orig_rv_part1, _ = self.set_periodicAlgebraicSystem(self.dof, orig_rv_part0, orig_rv_part1)

        uh_part0[:] = spsolve(orgPeriodicM, orig_rv_part0)
        uh_part1[:] = spsolve(orgPeriodicM, orig_rv_part1)

    def set_periodic_edge(self):
        """
        :return: idxPeriodicEdge0, 表示区域网格 `左侧` 的边
                 idxPeriodicEdge1, 表示区域网格 `右侧` 的边
                 idxNotPeriodicEdge, 表示区域网格 `上下两侧` 的边
        """
        mesh = self.mesh
        idxBdEdge = self.bdIndx

        mid_coor = mesh.entity_barycenter('edge')  # (NE,2)
        bd_mid = mid_coor[idxBdEdge, :]

        isPeriodicEdge0 = np.abs(bd_mid[:, 0] - 0.0) < 1e-8
        isPeriodicEdge1 = np.abs(bd_mid[:, 0] - 1.0) < 1e-8
        notPeriodicEdge = ~(isPeriodicEdge0 + isPeriodicEdge1)
        idxPeriodicEdge0 = idxBdEdge[isPeriodicEdge0]  # (NE_Peri,)
        idxPeriodicEdge1 = idxBdEdge[isPeriodicEdge1]  # (NE_Peri,)

        # |--- 检验 idxPeriodicEdge0 与 idxPeriodicEdge1 是否是一一对应的
        y_0 = mid_coor[idxPeriodicEdge0, 1]
        y_1 = mid_coor[idxPeriodicEdge1, 1]
        if np.allclose(np.sort(y_0), np.sort(y_1)) is False:
            raise ValueError("`idxPeriodicEdge0` and `idxPeriodicEdge1` are not 1-to-1.")
        idxNotPeriodicEdge = idxBdEdge[notPeriodicEdge]
        return idxPeriodicEdge0, idxPeriodicEdge1, idxNotPeriodicEdge

    def set_boundaryDofs(self, dof):
        """
        :param dof: dof, CH using the 'p'-order element
                |___ vdof, NS (velocity) using the 'p+1'-order element
        :return: periodicDof0, 表示区域网格 `左侧` 的边的自由度
                 periodicDof1, 表示区域网格 `右侧` 的边的自由度
                 np.unique(notPeriodicDof), 表示区域网格 `上下两侧` 的边的自由度
        """

        edge2dof = dof.edge_to_dof()
        periodicDof0 = edge2dof[self.idxPeriodicEdge0, :].flatten()
        periodicDof1 = edge2dof[self.idxPeriodicEdge1, :].flatten()
        notPeriodicDof = edge2dof[self.idxNotPeriodicEdge, :].flatten()

        # |--- 下面将左右两边的周期边界上的自由度一一对应起来
        periodicDof0 = np.setdiff1d(periodicDof0, notPeriodicDof)  # 返回的结果会去掉重复的项, 并自动从小到大排序
        periodicDof1 = np.setdiff1d(periodicDof1, notPeriodicDof)  # 返回的结果会去掉重复的项, 并自动从小到大排序
        #     |___ 注意, 上面这种处理方式直接将 `长方形` 区域的 `四个角点` 当做 Dirichlet-dof,
        #         |___ 即, 下面的 periodicDof0, periodicDof1 中是不包含区域的 `四个角点` 的.

        ip = dof.interpolation_points()  # 插值点, 也就是自由度所在的坐标
        ip_coory = ip[periodicDof0, 1]  # 得到 y 的坐标
        argsort = np.argsort(ip_coory)  # ip_coory 从小到大排序时, 返回原来的索引位置
        periodicDof0 = periodicDof0[argsort]  # 重新排列自由度
        ip_coory = ip[periodicDof1, 1]  # 得到 y 的坐标
        argsort = np.argsort(ip_coory)  # ip_coory 从小到大排序时, 返回原来的索引位置
        periodicDof1 = periodicDof1[argsort]  # 重新排列自由度

        if np.allclose(ip[periodicDof0, 1], ip[periodicDof1, 1]) is False:
            raise ValueError("`periodicDof0` and `periodicDof1` are not 1-to-1.")
        return periodicDof0, periodicDof1, np.unique(notPeriodicDof)

    def set_NS_Dirichlet_edge(self, idxDirEdge=None):
        if hasattr(self, 'idxNotPeriodicEdge'):
            idxDirEdge = self.idxNotPeriodicEdge
        else:
            self.idxPeriodicEdge0, self.idxPeriodicEdge1, self.idxNotPeriodicEdge = self.set_periodic_edge()
            idxDirEdge = self.idxNotPeriodicEdge
        return idxDirEdge

    def set_CH_Neumann_edge(self, idxNeuEdge=None):
        if hasattr(self, 'idxNotPeriodicEdge'):
            idxNeuEdge = self.idxNotPeriodicEdge
        else:
            self.idxPeriodicEdge0, self.idxPeriodicEdge1, self.idxNotPeriodicEdge = self.set_periodic_edge()
            idxNeuEdge = self.idxNotPeriodicEdge
        return idxNeuEdge

    def set_periodicAlgebraicSystem(self, dof, rhsVec0, rhsVec1, lhsM=None):
        if dof.p == self.p:
            periodicDof0, periodicDof1 = self.periodicDof0, self.periodicDof1
            #   |___ the phi_h- and p_h-related variables periodic dofs (using p-order polynomial)
        else:
            periodicDof0, periodicDof1 = self.vPeriodicDof0, self.vPeriodicDof1
            #   |___ the velocity-related variables periodic dofs (using (p+1)-order polynomial)

        NPDof = len(periodicDof0)
        NglobalDof = len(rhsVec0)
        IM = identity(NglobalDof, dtype=np.int_)
        if lhsM is None:
            lhsM = IM.copy()

        oneVec = np.ones((NPDof,), dtype=np.int_)
        operatorM = IM + csr_matrix((oneVec, (periodicDof0, periodicDof1)), shape=(NglobalDof, NglobalDof), dtype=np.int_)
        lhsM = operatorM @ lhsM
        rhsVec0 = operatorM @ rhsVec0
        rhsVec1 = operatorM @ rhsVec1
        #    |___ operatorM@lhsM: lhsM 中第 periodicDof1 行加到 periodicDof0 上; rhsVec0, rhsVec1 同理.

        oneVec = np.ones((NglobalDof,), dtype=np.int_)
        oneVec[periodicDof1] = np.int_(0)
        operatorM = spdiags(oneVec, 0, NglobalDof, NglobalDof)
        lhsM = operatorM @ lhsM
        rhsVec0 = operatorM @ rhsVec0
        rhsVec1 = operatorM @ rhsVec1
        #    |___ rowM0@lhsM: lhsM 中第 periodicDof1 行所有元素设置为 0; rhsVec0, rhsVec1 同理.

        oneVec = np.ones((2 * NPDof,), dtype=np.int_)
        oneVec[NPDof:] = np.int_(-1)
        operatorM = csr_matrix((oneVec, (np.hstack([periodicDof1, periodicDof1]), np.hstack([periodicDof0, periodicDof1]))),
                               shape=(NglobalDof, NglobalDof), dtype=np.int_)
        lhsM += operatorM
        #    |___ lhsM 中第 periodicDof1 行: 第 periodicDof0 列为 1, 第 periodicDof1 列为 -1.

        return rhsVec0, rhsVec1, lhsM

