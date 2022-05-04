#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: CoCurrentFlowModel2d.py
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
from sympy import symbols, tanh, lambdify
from scipy.sparse import csr_matrix, spdiags, identity, eye, bmat
import pickle
# from fealpy.quadrature import FEMeshIntegralAlg
from scipy.sparse.linalg import spsolve
from fealpy.boundarycondition import DirichletBC
# from fealpy.decorator import timer
# from fealpy.functionspace import LagrangeFiniteElementSpace
# from sym_diff_basis import compute_basis
from FEM_CH_NS_Model2d import FEM_CH_NS_Model2d
import matplotlib.pyplot as plt


class CoCurrentFlowModel2d(FEM_CH_NS_Model2d):
    """
        注意:
            本程序所参考的数值格式是按照 $D(u)=\nabla u + \nabla u^T$ 来写的,
            如果要调整为 $D(u)=(\nabla u + \nabla u^T)/2$, 需要调整的地方太多了,
            所以为了保证和数值格式的一致性, 本程序也是严格按照 $D(u)=\nabla u + \nabla u^T$ 来编写的.
        """

    def __init__(self, pde, mesh, p, dt):
        super(CoCurrentFlowModel2d, self).__init__(pde, mesh, p, dt)
        self.wh_part0 = self.space.function()
        self.wh_part1 = self.space.function()
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

        self.grad_mu_val = np.empty((2, 2))  # 此项在 `update_mu_and_Xi()` 中更新.
        self.rho_bar_n = 0.  # 此项在 `decoupled_NS_addXi_Solver_T1stOrder()` 中更新: 为了获得第 n 时间层的取值 (在 `update_mu_and_Xi()` 会用到).
        self.nu_bar_n = 0.  # 此项在 `decoupled_NS_addXi_Solver_T1stOrder()` 中更新: 为了获得第 n 时间层的取值 (在 `update_mu_and_Xi()` 会用到).
        self.R_n = 1.  # 此项在 `update_mu_and_Xi()` 中更新.
        self.C0 = 1.  # 此项在 `update_mu_and_Xi()` 中, 以保证 E_n = \int H(\phi) + C0 > 0.
        self.Xi = 1.  # 此项在 `update_mu_and_Xi()` 中更新.

        if hasattr(self, 'idxNotPeriodicEdge') is False:
            self.idxPeriodicEdge0, self.idxPeriodicEdge1, self.idxNotPeriodicEdge = self.set_periodic_edge()

        # |--- setting periodic dofs
        self.periodicDof0, self.periodicDof1, self.notPeriodicDof = self.set_boundaryDofs(self.dof)
        #   |___ the phi_h- and p_h-related variables periodic dofs (using p-order polynomial)
        self.vPeriodicDof0, self.vPeriodicDof1, self.vNotPeriodicDof = self.set_boundaryDofs(self.vdof)
        #   |___ the velocity-related variables periodic dofs (using (p+1)-order polynomial)

        # |--- NS: setting algebraic system for periodic boundary condition
        self.plsm = 1. / min(self.pde.rho0, self.pde.rho1) * self.StiffMatrix
        self.pPeriodicM_NS = None

        self.auxVLM = 1. / self.dt * self.vel_MM
        self.vAuxPeriodicM_NS = None

        self.VLM = 1. / self.dt * self.vel_MM + max(self.pde.nu0 / self.pde.rho0, self.pde.nu1 / self.pde.rho1) * self.vel_SM
        self.vOrgPeriodicM_NS = None

        self.number_save = 150

    def set_CH_Coeff(self, dt_minimum=None):
        """
        This function is designed to pass the father-function in 'FEM_CH_NS_Model2d.py'
        :param dt_minimum:
        :return:
        """
        s = 0
        alpha = 0
        return s, alpha

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
                return pde.solution_CH(p)
            uh[:] = self.space.interpolation(init_solution_CH)

            def init_velocity0(p):
                return 0. * p[..., 0]
            vel0[:] = self.vspace.interpolation(init_velocity0)

            def init_velocity1(p):
                return 0. * p[..., 0]
            vel1[:] = self.vspace.interpolation(init_velocity1)

            ph[:] = self.get_init_pressure()

        # # time-looping
        print('    # ------------ begin the time-looping ------------ #')
        v_ip_coord = self.vspace.interpolation_points()  # (vNdof,2)
        val0_at_0 = np.zeros([len(self.vPeriodicDof0) + 2, 2], dtype=np.float)
        filename_basic = ('./CoCurrentFlowOutput/' + 'CCF_T(' + str(self.pde.T) + ')_dt(' + ('%.e' % dt) + ')_eta('
                          + ('%.e' % self.pde.eta) + ')')
        for nt in range(NT - 1):
            currt_t = timemesh[nt]
            next_t = currt_t + dt
            self.decoupled_NS_addXi_Solver_T1stOrder(vel0, vel1, ph, uh, next_t)
            Xi = self.update_mu_and_Xi(uh, next_t)

            # |--- update the values
            wh[:] = self.wh_part0[:] + Xi * self.wh_part1[:]
            ph[:] = self.ph_part0[:] + Xi * self.ph_part1[:]
            vel0[:] = self.vel0_part0[:] + Xi * self.vel0_part1[:]
            vel1[:] = self.vel1_part0[:] + Xi * self.vel1_part1[:]
            # print('    end of one-looping')

            if nt % max([int(NT / self.number_save), 1]) == 0:
                print('    currt_t = %.4e' % currt_t)
                filename = filename_basic + '_nt(' + str(nt) + ')'
                val0_at_0[1:-1, 1] = vel0[self.vPeriodicDof0]
                val0_at_0[1:-1, 0] = v_ip_coord[self.vPeriodicDof0, 1]
                val0_at_0[0, 0] = -1.
                val0_at_0[-1, 0] = 1.
                true_solution = self.plot_true_solution(val0_at_0[:, 0])

                plt.figure()
                plt.plot(val0_at_0[:, 0], val0_at_0[:, 1], color='b', linewidth=0.6, label='Numerical')
                plt.plot(val0_at_0[:, 0], true_solution, color='r', linewidth=0.6, label='True')
                plt.xlabel("Y")
                plt.ylabel("axial velocity")
                plt.legend(loc=4, fontsize='8')
                plt.savefig(filename + '.png')
                plt.close()

                programData = {'nt': nt, 'uh': uh, 'vel0': vel0, 'vel1': vel1, 'ph': ph, 'val0_at_0': val0_at_0}
                self.pickle_save_data(filename, programData)

                # |--- compute errs
                uh_l2err = self.space.integralalg.L2_error(pde.zero_func, uh)
                v0_l2err_NS = self.vspace.integralalg.L2_error(pde.zero_func, vel0)
                v1_l2err_NS = self.vspace.integralalg.L2_error(pde.zero_func, vel1)
                ph_l2err = self.space.integralalg.L2_error(pde.zero_func, ph)
                vel_l2err = np.sqrt(v0_l2err_NS**2 + v1_l2err_NS**2)
                if np.isnan(uh_l2err) | np.isnan(vel_l2err) | np.isnan(ph_l2err):
                    print('Some error is nan: breaking the program')
                    break
        print('    # ------------ end the time-looping ------------ #\n')

        filename = filename_basic + '_nt(' + str(NT - 1) + ')'
        val0_at_0[1:-1, 1] = vel0[self.vPeriodicDof0]
        val0_at_0[1:-1, 0] = v_ip_coord[self.vPeriodicDof0, 1]
        val0_at_0[0, 0] = -1.
        val0_at_0[-1, 0] = 1.
        true_solution = self.plot_true_solution(val0_at_0[:, 0])

        plt.figure()
        plt.plot(val0_at_0[:, 0], val0_at_0[:, 1], color='b', linewidth=0.6, label='Numerical')
        plt.plot(val0_at_0[:, 0], true_solution, color='r', linewidth=0.6, label='True')
        plt.xlabel("Y")
        plt.ylabel("axial velocity")
        plt.legend(loc=4, fontsize='8')
        plt.savefig(filename + '.png')
        plt.close()

        programData = {'nt': NT-1, 'uh': uh, 'vel0': vel0, 'vel1': vel1, 'ph': ph, 'val0_at_0': val0_at_0}
        self.pickle_save_data(filename, programData)

        return val0_at_0

    def restart_CH_NS_addXi_Solver_T1stOrder(self, load_filename):
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

        program_data = self.pickle_load_data(load_filename)
        nt_ = program_data['nt']
        uh[:] = program_data['uh'][:]
        vel0[:] = program_data['vel0'][:]
        vel1[:] = program_data['vel1'][:]
        ph[:] = program_data['ph'][:]

        # # time-looping
        print('    # ------------ begin the time-looping ------------ #')
        v_ip_coord = self.vspace.interpolation_points()  # (vNdof,2)
        val0_at_0 = np.zeros([len(self.vPeriodicDof0) + 2, 2], dtype=np.float)
        filename_basic = ('./CoCurrentFlowOutput/' + 'CCF_T(' + str(self.pde.T) + ')_dt(' + ('%.e' % dt) + ')_eta('
                          + ('%.e' % self.pde.eta) + ')')
        for nt in range(nt_ + 1, NT - 1):
            currt_t = timemesh[nt]
            next_t = currt_t + dt
            self.decoupled_NS_addXi_Solver_T1stOrder(vel0, vel1, ph, uh, next_t)
            Xi = self.update_mu_and_Xi(uh, next_t)

            # |--- update the values
            wh[:] = self.wh_part0[:] + Xi * self.wh_part1[:]
            ph[:] = self.ph_part0[:] + Xi * self.ph_part1[:]
            vel0[:] = self.vel0_part0[:] + Xi * self.vel0_part1[:]
            vel1[:] = self.vel1_part0[:] + Xi * self.vel1_part1[:]
            # print('    end of one-looping')

            if nt % max([int(NT / self.number_save), 1]) == 0:
                print('    currt_t = %.4e' % currt_t)
                filename = filename_basic + '_nt(' + str(nt) + ')'
                val0_at_0[1:-1, 1] = vel0[self.vPeriodicDof0]
                val0_at_0[1:-1, 0] = v_ip_coord[self.vPeriodicDof0, 1]
                val0_at_0[0, 0] = -1.
                val0_at_0[-1, 0] = 1.
                true_solution = self.plot_true_solution(val0_at_0[:, 0])

                plt.figure()
                plt.plot(val0_at_0[:, 0], val0_at_0[:, 1], color='b', linewidth=0.6, label='Numerical')
                plt.plot(val0_at_0[:, 0], true_solution, color='r', linewidth=0.6, label='True')
                plt.xlabel("Y")
                plt.ylabel("axial velocity")
                plt.legend(loc=4, fontsize='8')
                plt.savefig(filename + '.png')
                plt.close()

                programData = {'nt': nt, 'uh': uh, 'vel0': vel0, 'vel1': vel1, 'ph': ph, 'val0_at_0': val0_at_0}
                self.pickle_save_data(filename, programData)

                # |--- compute errs
                uh_l2err = self.space.integralalg.L2_error(pde.zero_func, uh)
                v0_l2err_NS = self.vspace.integralalg.L2_error(pde.zero_func, vel0)
                v1_l2err_NS = self.vspace.integralalg.L2_error(pde.zero_func, vel1)
                ph_l2err = self.space.integralalg.L2_error(pde.zero_func, ph)
                vel_l2err = np.sqrt(v0_l2err_NS**2 + v1_l2err_NS**2)
                if np.isnan(uh_l2err) | np.isnan(vel_l2err) | np.isnan(ph_l2err):
                    print('Some error is nan: breaking the program')
                    break
        print('    # ------------ end the time-looping ------------ #\n')

        filename = filename_basic + '_nt(' + str(NT - 1) + ')'
        val0_at_0[1:-1, 1] = vel0[self.vPeriodicDof0]
        val0_at_0[1:-1, 0] = v_ip_coord[self.vPeriodicDof0, 1]
        val0_at_0[0, 0] = -1.
        val0_at_0[-1, 0] = 1.
        true_solution = self.plot_true_solution(val0_at_0[:, 0])

        plt.figure()
        plt.plot(val0_at_0[:, 0], val0_at_0[:, 1], color='b', linewidth=0.6, label='Numerical')
        plt.plot(val0_at_0[:, 0], true_solution, color='r', linewidth=0.6, label='True')
        plt.xlabel("Y")
        plt.ylabel("axial velocity")
        plt.legend(loc=4, fontsize='8')
        plt.savefig(filename + '.png')
        plt.close()

        programData = {'nt': NT-1, 'uh': uh, 'vel0': vel0, 'vel1': vel1, 'ph': ph, 'val0_at_0': val0_at_0}
        self.pickle_save_data(filename, programData)

        return val0_at_0

    def get_init_pressure(self):
        plsm = self.space.stiff_matrix()
        f_val_NS = self.pde.source_NS(self.c_pp)  # (NQ,NC,GD)
        cell_int = np.einsum('i, ijm, ijkm, j->jk', self.c_ws, f_val_NS, self.gphi_c, self.cellmeasure)  # (NC,ldof)
        prv = np.zeros((self.dof.number_of_global_dofs(),), dtype=self.ftype)  # (Npdof,)
        np.add.at(prv, self.cell2dof, cell_int)

        basis_int = self.space.integral_basis()
        plsm_temp = bmat([[plsm, basis_int.reshape(-1, 1)], [basis_int, None]], format='csr')
        prv = np.r_[prv, 0]
        ph = spsolve(plsm_temp, prv)[:-1]  # we have added one additional dof
        return ph

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
        pass

    def pickle_save_data(self, filename, data):
        file_point = open(filename + '.pkl', 'wb')
        file_point.write(pickle.dumps(data))
        file_point.close()

    def pickle_load_data(self, filename):
        file_point = open(filename + '.pkl', 'rb')
        data = pickle.loads(file_point.read())
        file_point.close()
        return data

    def plot_true_solution(self, x_coord):
        KK = symbols('K')
        t, x, y, pi = symbols('t x y pi')
        epsilon, m = symbols('epsilon m')

        pde = self.pde
        rho0 = pde.rho0
        rho1 = pde.rho1
        nu0 = pde.nu0
        nu1 = pde.nu1
        r0 = pde.r0
        r1 = pde.r1
        eta = pde.eta

        # |--- 2D-par-setting
        n = 0
        R = y / r1
        nu_hat = nu1 / nu0
        delta = r0 / r1
        C = (n + 3.) / 2

        # |--- CH
        u = -tanh((y - r0) / (np.sqrt(2) * eta))
        rho = (rho0 + rho1) / 2 + (rho0 - rho1) / 2 * u
        nu = (nu0 + nu1) / 2 + (nu0 - nu1) / 2 * u

        # |--- NS
        vel_bar = KK * r1 ** 2 / (nu1 * (n + 1) * (n + 3)) * (delta ** (n + 3) * (nu_hat - 1) + 1)
        vel0_domain0 = vel_bar * C * (1 - delta ** 2 + nu_hat * (delta ** 2 - R ** 2)) / (delta ** (n + 3) * (nu_hat - 1) + 1)
        vel0_domain1 = vel_bar * C * (1 - R ** 2) / (delta ** (n + 3) * (nu_hat - 1) + 1)

        vel0_domain0_f = lambdify([y, KK], vel0_domain0, "numpy")
        vel0_domain1_f = lambdify([y, KK], vel0_domain1, "numpy")

        # |--- setting the demarcation point
        dem_point0, = np.nonzero(abs(x_coord - r0) < 1.e-9)
        dem_point1, = np.nonzero(abs(x_coord - (-r0)) < 1.e-9)

        if (any(dem_point0) and any(dem_point1)) is False:
            raise ValueError("There is no `dem_point0` or `dem_point1`")
        x_domain1_0 = x_coord[:dem_point0]
        x_domain0 = x_coord[dem_point0:dem_point1]
        x_domain1_1 = x_coord[dem_point1:]

        K = pde.K
        y_domain1_0 = vel0_domain1_f(x_domain1_0, K)
        y_domain0 = vel0_domain0_f(x_domain0, K)
        y_domain1_1 = vel0_domain1_f(x_domain1_1, K)
        yy = np.concatenate([np.concatenate([y_domain1_0, y_domain0]), y_domain1_1])
        return yy

    def decoupled_NS_addXi_Solver_T1stOrder(self, vel0, vel1, ph, uh, next_t):
        """
        The decoupled-Navier-Stokes-solver for the all system.
        :param vel0: The fist-component of velocity: stored the n-th(time) value, and to update the (n+1)-th value.
        :param vel1: The second-component of velocity: stored the n-th(time) value, and to update the (n+1)-th value.
        :param ph: The pressure: stored the n-th(time) value, and to update the (n+1)-th value.
        :param uh: The n-th(time) value of the solution of Cahn-Hilliard equation.
        :param next_t: The next-time.
        :return: Updated vel0, vel1, ph.

        注意:
            本程序所参考的数值格式是按照 $D(u)=\nabla u + \nabla u^T$ 来写的,
            如果要调整为 $D(u)=(\nabla u + \nabla u^T)/2$, 需要调整的地方太多了,
            所以为了保证和数值格式的一致性, 本程序也是严格按照 $D(u)=\nabla u + \nabla u^T$ 来编写的.
        """

        ph_part0 = self.ph_part0
        ph_part1 = self.ph_part1
        vel0_part0 = self.vel0_part0
        vel1_part0 = self.vel1_part0
        vel0_part1 = self.vel0_part1
        vel1_part1 = self.vel1_part1
        auxVel0_part0 = self.auxVel0_part0
        auxVel1_part0 = self.auxVel1_part0
        auxVel0_part1 = self.auxVel0_part1
        auxVel1_part1 = self.auxVel1_part1

        # |--- variable coefficients settings
        pde = self.pde
        m = pde.m
        rho0 = pde.rho0
        rho1 = pde.rho1
        nu0 = pde.nu0
        nu1 = pde.nu1

        rho_min = min(rho0, rho1)
        eta_max = max(nu0 / rho0, nu1 / rho1)
        J0 = -1. / 2 * (rho0 - rho1) * m

        nDir_NS = self.nDir_NS  # (NDir,GD), here, the GD is 2
        nbdEdge = self.nbdEdge

        # |--- grad-velocity at Dirichlet boundary
        grad_vel0_f = self.uh_grad_value_at_faces(vel0, self.f_bcs, self.DirCellIdx_NS, self.DirLocalIdx_NS,
                                                  space=self.vspace)  # grad_vel0: (NQ,NDir,GD)
        grad_vel1_f = self.uh_grad_value_at_faces(vel1, self.f_bcs, self.DirCellIdx_NS, self.DirLocalIdx_NS,
                                                  space=self.vspace)  # grad_vel1: (NQ,NDir,GD)

        # |--- for cell-integration
        # |--- ph, uh, vel0, vel1 are the (n)-th time step values
        grad_ph_val = self.space.grad_value(ph, self.c_bcs)  # (NQ,NC,GD)
        uh_val = self.space.value(uh, self.c_bcs)  # (NQ,NC)
        uh_val_f = self.space.value(uh, self.f_bcs)[..., self.DirEdgeIdx_NS]  # (NQ,NDir)
        #    |___ TODO: 为什么是 [..., self.DirCellIdx_NS] 而不是 [..., self.DirEdgeIdx_NS]?? 应该是个 bug ??
        #              |___ TODO: 2022-03-08 改为 [..., self.DirEdgeIdx_NS]
        grad_uh_val = self.space.grad_value(uh, self.c_bcs)  # (NQ,NC,GD)
        vel0_val = self.vspace.value(vel0, self.c_bcs)  # (NQ,NC)
        vel1_val = self.vspace.value(vel1, self.c_bcs)  # (NQ,NC)
        grad_vel0_val = self.vspace.grad_value(vel0, self.c_bcs)  # (NQ,NC,GD)
        grad_vel1_val = self.vspace.grad_value(vel1, self.c_bcs)  # (NQ,NC,GD)

        nolinear_val = self.NSNolinearTerm(vel0, vel1, self.c_bcs)  # (NQ,NC,GD)
        velDir_val = pde.dirichlet_NS(self.f_pp_Dir_NS, next_t)  # (NQ,NDir,GD)
        f_val_NS = pde.source_NS(self.c_pp)  # (NQ,NC,GD)

        # |--- grad-free-energy: \nabla h(phi)
        #     |___ where h(phi) = epsilon/eta^2*phi*(phi^2-1)
        grad_free_energy_c = pde.epsilon / pde.eta ** 2 * self.grad_free_energy_at_cells(uh, self.c_bcs)  # (NQ,NC,2)

        # |--- the CH_term: uh_val * (-epsilon*\nabla\Delta uh_val + \nabla free_energy)
        if self.p < 3:
            grad_x_laplace_uh = np.zeros(grad_free_energy_c[..., 0].shape)
            grad_y_laplace_uh = np.zeros(grad_free_energy_c[..., 0].shape)
        elif self.p == 3:
            phi_xxx, phi_yyy, phi_yxx, phi_xyy = self.cb.get_highorder_diff(self.c_bcs, order='3rd-order')  # (NQ,NC,ldof)
            grad_x_laplace_uh = -pde.epsilon * np.einsum('ijk, jk->ij', phi_xxx + phi_xyy, uh[self.cell2dof])  # (NQ,NC)
            grad_y_laplace_uh = -pde.epsilon * np.einsum('ijk, jk->ij', phi_yxx + phi_yyy, uh[self.cell2dof])  # (NQ,NC)
        else:
            raise ValueError("The polynomial order p should be <= 3.")

        # |--- Method II: 此处, 既然 phi^n 和 phi^{n-1} 都是已知的,
        #                 |___ 那么直接利用 mu^n=-lambda\Delta(phi^n)+h(phi^{n-1}), 计算 \nabla(mu^n).
        grad_mu_val = np.array([grad_x_laplace_uh + grad_free_energy_c[..., 0],
                                grad_y_laplace_uh + grad_free_energy_c[..., 1]]).transpose([1, 2, 0])  # (NQ,NC,2)
        self.grad_mu_val = grad_mu_val

        # |--- update the variable coefficients
        self.rho_bar_n = (rho0 + rho1) / 2. + (rho0 - rho1) / 2. * uh_val  # (NQ,NC)
        self.nu_bar_n = (nu0 + nu1) / 2. + (nu0 - nu1) / 2. * uh_val  # (NQ,NC)
        rho_bar_n = self.rho_bar_n
        nu_bar_n = self.nu_bar_n
        rho_bar_n_f = (rho0 + rho1) / 2. + (rho0 - rho1) / 2. * uh_val_f  # (NQ,NDir)
        nu_bar_n_f = (nu0 + nu1) / 2. + (nu0 - nu1) / 2. * uh_val_f  # (NQ,NDir)
        J_n = J0 * grad_mu_val  # (NQ,NC,GD)
        eta_n = nu_bar_n / rho_bar_n  # (NQ,NC)
        eta_n_f = nu_bar_n_f / rho_bar_n_f  # (NQ,NDir)
        eta_nx = (((nu0 - nu1) / 2. * rho_bar_n - (rho0 - rho1) / 2. * nu_bar_n)
                  * grad_uh_val[..., 0] / rho_bar_n ** 2)  # (NQ,NC)
        eta_ny = (((nu0 - nu1) / 2. * rho_bar_n - (rho0 - rho1) / 2. * nu_bar_n)
                  * grad_uh_val[..., 1] / rho_bar_n ** 2)  # (NQ,NC)

        # |--- the auxiliary variable: G_VC
        vel_stress_mat = [[2 * grad_vel0_val[..., 0], (grad_vel0_val[..., 1] + grad_vel1_val[..., 0])],
                          [(grad_vel0_val[..., 1] + grad_vel1_val[..., 0]), 2 * grad_vel1_val[..., 1]]]
        vel_grad_mat = [[grad_vel0_val[..., 0], grad_vel0_val[..., 1]],
                        [grad_vel1_val[..., 0], grad_vel1_val[..., 1]]]
        rho_bar_n_axis = rho_bar_n[..., np.newaxis]
        G_VC = 1. / rho_bar_n_axis * (
                -rho_bar_n_axis * nolinear_val - self.vec_div_mat(J_n, vel_grad_mat)
                + self.vec_div_mat((nu0 - nu1) / 2. * grad_uh_val, vel_stress_mat))  # (NQ,NC,2)

        curl_vel = grad_vel1_val[..., 0] - grad_vel0_val[..., 1]  # (NQ,NC)
        curl_vel_f = grad_vel1_f[..., 0] - grad_vel0_f[..., 1]  # (NQ,NDir)

        # |--- Dirichlet faces integration of 1/dt*(auxVel, \nabla q)_\Omega = -1/dt*<vel\cdot n, q>_\Gamma for part0
        dir_int0_forpart0 = -1 / self.dt * np.einsum('i, ijk, jk, ijn, j->jn', self.f_ws, velDir_val, nDir_NS, self.phi_f,
                                                     self.DirEdgeMeasure_NS)  # (NDir,fldof)
        # |--- Dirichlet faces integration of -<eta_n * n x curl_vel, \nabla q>_\Gamma =
        dir_int1_forpart1 = -(np.einsum('i, j, ij, jin, j->jn', self.f_ws, nDir_NS[:, 1], eta_n_f * curl_vel_f, self.gphi_f[..., 0],
                                        self.DirEdgeMeasure_NS)
                              + np.einsum('i, j, ij, jin, j->jn', self.f_ws, -nDir_NS[:, 0], eta_n_f * curl_vel_f, self.gphi_f[..., 1],
                                          self.DirEdgeMeasure_NS))  # (NDir,cldof)

        # for cell integration
        temp = (1. / rho_bar_n_axis * f_val_NS + 1. / self.dt * np.array([vel0_val, vel1_val]).transpose((1, 2, 0))
                + (1. / rho_min - 1. / rho_bar_n_axis) * grad_ph_val)  # (NQ,NC,2)
        cell_int0_forpart0 = np.einsum('i, ijs, ijks, j->jk', self.c_ws, temp, self.gphi_c, self.cellmeasure)  # (NC,cldof)
        cell_int0_forpart1 = np.einsum('i, ijs, ijks, j->jk', self.c_ws, G_VC, self.gphi_c, self.cellmeasure)  # (NC,cldof)
        cell_int1_forpart1 = (np.einsum('i, ij, ijk, j->jk', self.c_ws, eta_ny * curl_vel, self.gphi_c[..., 0], self.cellmeasure)
                              + np.einsum('i, ij, ijk, j->jk', self.c_ws, -eta_nx * curl_vel, self.gphi_c[..., 1], self.cellmeasure))  # (NC,cldof)

        # # ------------------------------------ # #
        # # --- to update the pressure value --- # #
        # # ------------------------------------ # #
        # |--- 1. assemble the NS's pressure equation
        prv_part0 = np.zeros((self.dof.number_of_global_dofs(),), dtype=self.ftype)  # (Npdof,) the Pressure's Right-hand Vector
        np.add.at(prv_part0, self.cell2dof, cell_int0_forpart0)
        np.add.at(prv_part0, self.face2dof[self.DirEdgeIdx_NS, :], dir_int0_forpart0)

        prv_part1 = np.zeros((self.dof.number_of_global_dofs(),), dtype=self.ftype)  # (Npdof,) the Pressure's Right-hand Vector
        np.add.at(prv_part1, self.cell2dof, cell_int0_forpart1 + cell_int1_forpart1)
        np.add.at(prv_part1, self.cell2dof[self.DirCellIdx_NS, :], dir_int1_forpart1)

        # |--- 2. solve the NS's pressure equation
        plsm = 1. / rho_min * self.StiffMatrix

        # |--- 3. Method I: The following code is right! Pressure satisfies \int_\Omega p = 0
        #      |___ TODO: 检验这个处理 pressure 的方式是否正确?
        prv_part0 = np.r_[prv_part0, 0]
        prv_part1 = np.r_[prv_part1, 0]
        if self.pPeriodicM_NS is None:
            basis_int = self.space.integral_basis()
            plsm_temp = bmat([[plsm, basis_int.reshape(-1, 1)], [basis_int, None]], format='csr')
            prv_part0, prv_part1, pPeriodicM = self.set_periodicAlgebraicSystem(self.dof, prv_part0, prv_part1, plsm_temp)
            self.pPeriodicM_NS = pPeriodicM.copy()
        else:
            pPeriodicM = self.pPeriodicM_NS
            prv_part0, prv_part1, _ = self.set_periodicAlgebraicSystem(self.dof, prv_part0, prv_part1)
        ph_part0[:] = spsolve(pPeriodicM, prv_part0)[:-1]  # we have added one additional dof
        ph_part1[:] = spsolve(pPeriodicM, prv_part1)[:-1]  # we have added one additional dof

        # # ---------------------------------------- # #
        # # --- to update the aux-velocity value --- # #
        # # ---------------------------------------- # #
        grad_ph_part0_val = self.space.grad_value(ph_part0, self.c_bcs)  # (NQ,NC,2)
        grad_ph_part1_val = self.space.grad_value(ph_part1, self.c_bcs)  # (NQ,NC,2)

        # |--- the aux-Velocity-Left-Matrix
        auxVLM = 1. / self.dt * self.vel_MM

        # |--- assemble the first-component of Velocity-Right-Vector
        vel_val = [vel0_val, vel1_val]
        mask_eta_n = [eta_ny, -eta_nx]

        def solve_auxVel_part0(whichIdx):
            auxVRV = np.zeros((self.vdof.number_of_global_dofs(),), dtype=self.ftype)  # (Nvdof,)
            VRVtemp = (1. / rho_bar_n * f_val_NS[..., whichIdx] + 1. / self.dt * vel_val[whichIdx]
                       - 1. / rho_min * grad_ph_part0_val[..., whichIdx]
                       + (1. / rho_min - 1. / rho_bar_n) * grad_ph_val[..., whichIdx])  # (NQ,NC)
            cellInt = np.einsum('i, ij, ijk, j->jk', self.c_ws, VRVtemp, self.vphi_c, self.cellmeasure)  # (NC,clodf)
            np.add.at(auxVRV, self.vcell2dof, cellInt)
            if self.vAuxPeriodicM_NS is None:
                auxVRV, _, vAuxPeriodicM = self.set_periodicAlgebraicSystem(self.vdof, auxVRV, auxVRV.copy(), auxVLM)
                self.vAuxPeriodicM_NS = vAuxPeriodicM.copy()
            else:
                vAuxPeriodicM = self.vAuxPeriodicM_NS
                auxVRV, _, _ = self.set_periodicAlgebraicSystem(self.vdof, auxVRV, auxVRV.copy())
            return spsolve(vAuxPeriodicM, auxVRV).reshape(-1)
        auxVel0_part0[:] = solve_auxVel_part0(0)
        auxVel1_part0[:] = solve_auxVel_part0(1)

        def solve_auxVel_part1(whichIdx):
            mask_Idx = np.mod(whichIdx + 1, 2)
            auxVRV = np.zeros((self.vdof.number_of_global_dofs(),), dtype=self.ftype)  # (Nvdof,)
            VRVtemp = (-1. / rho_min * grad_ph_part1_val[..., whichIdx] + G_VC[..., whichIdx] + mask_eta_n[whichIdx] * curl_vel)  # (NQ,NC)
            cellInt0 = np.einsum('i, ij, ijk, j->jk', self.c_ws, VRVtemp, self.vphi_c, self.cellmeasure)  # (NC,clodf)
            cellInt1 = np.einsum('i, ij, ijk, j->jk', self.c_ws, eta_n * curl_vel,
                                 (-1) ** whichIdx * self.vgphi_c[..., mask_Idx], self.cellmeasure)  # (NC,clodf)
            edgeInt = -np.einsum('i, j, ij, ijn, j->jn', self.f_ws, (-1) ** whichIdx * nDir_NS[:, mask_Idx], eta_n_f * curl_vel_f,
                                 self.vphi_f, self.DirEdgeMeasure_NS)  # (NBE,vfldof)
            np.add.at(auxVRV, self.vcell2dof, cellInt0 + cellInt1)
            np.add.at(auxVRV, self.vface2dof[self.DirEdgeIdx_NS, :], edgeInt)

            vAuxPeriodicM = self.vAuxPeriodicM_NS
            auxVRV, _, _ = self.set_periodicAlgebraicSystem(self.vdof, auxVRV, auxVRV.copy())
            return spsolve(vAuxPeriodicM, auxVRV).reshape(-1)
        auxVel0_part1[:] = solve_auxVel_part1(0)
        auxVel1_part1[:] = solve_auxVel_part1(1)

        # # ------------------------------------ # #
        # # --- to update the velocity value --- # #
        # # ------------------------------------ # #
        # # the Velocity-Left-Matrix
        VLM = 1. / self.dt * self.vel_MM + eta_max * self.vel_SM

        def dir_u0(p):
            return pde.dirichlet_NS(p, next_t)[..., 0]

        def dir_u1(p):
            return pde.dirichlet_NS(p, next_t)[..., 1]
        dir_func = [dir_u0, dir_u1]

        def solve_Vel_part0(whichIdx):
            VRV = np.zeros((self.vdof.number_of_global_dofs(),), dtype=self.ftype)  # (Nvdof,)
            VRVtemp = (1. / rho_bar_n * f_val_NS[..., whichIdx] + 1. / self.dt * vel_val[whichIdx]
                       - 1. / rho_min * grad_ph_part0_val[..., whichIdx]
                       + (1. / rho_min - 1. / rho_bar_n) * grad_ph_val[..., whichIdx])  # (NQ,NC)
            cellInt = np.einsum('i, ij, ijk, j->jk', self.c_ws, VRVtemp, self.vphi_c, self.cellmeasure)  # (NC,clodf)
            np.add.at(VRV, self.vcell2dof, cellInt)
            V_BC = DirichletBC(self.vspace, dir_func[whichIdx], threshold=self.DirEdgeIdx_NS)
            VLM_Temp, VRV = V_BC.apply(VLM.copy(), VRV)
            VRV, _, vOrgPeriodicM = self.set_periodicAlgebraicSystem(self.vdof, VRV, VRV.copy(), VLM_Temp)
            return spsolve(vOrgPeriodicM, VRV).reshape(-1)
        vel0_part0[:] = solve_Vel_part0(0)
        vel1_part0[:] = solve_Vel_part0(1)

        def zero_func(p):
            # |--- to handle the zero-boundary-condition for velocity-part1 terms.
            return 0. * p[..., 0]

        def solve_Vel_part1(whichIdx):
            mask_Idx = np.mod(whichIdx + 1, 2)
            VRV = np.zeros((self.vdof.number_of_global_dofs(),), dtype=self.ftype)  # (Nvdof,)
            VRVtemp = (-1. / rho_min * grad_ph_part1_val[..., whichIdx] + G_VC[..., whichIdx] + mask_eta_n[whichIdx] * curl_vel)  # (NQ,NC)
            cellInt0 = np.einsum('i, ij, ijk, j->jk', self.c_ws, VRVtemp, self.vphi_c, self.cellmeasure)  # (NC,clodf)
            cellInt1 = np.einsum('i, ij, ijk, j->jk', self.c_ws, (eta_n - eta_max) * curl_vel,
                                 (-1) ** whichIdx * self.vgphi_c[..., mask_Idx], self.cellmeasure)  # (NC,clodf)
            np.add.at(VRV, self.vcell2dof, cellInt0 + cellInt1)
            V_BC = DirichletBC(self.vspace, zero_func, threshold=self.DirEdgeIdx_NS)
            VLM_Temp, VRV = V_BC.apply(VLM.copy(), VRV)
            VRV, _, vOrgPeriodicM = self.set_periodicAlgebraicSystem(self.vdof, VRV, VRV.copy(), VLM_Temp)
            return spsolve(vOrgPeriodicM, VRV).reshape(-1)
        vel0_part1[:] = solve_Vel_part1(0)
        vel1_part1[:] = solve_Vel_part1(1)

    def update_mu_and_Xi(self, uh, next_t):
        """
        Compute the discretization two parts of `mu = -epsilon *\\Delta\\phi + h(\\phi)` at cells' c_bcs.
        ---
        :param uh: the (n)-th time step value.
        :param next_t:
        :return:
        """

        space = self.space
        vspace = self.vspace
        c_bcs = self.c_bcs

        # |--- all the *_part* is the (n+1)-th time step values
        vel0_part0 = self.vel0_part0
        vel1_part0 = self.vel1_part0
        vel0_part1 = self.vel0_part1
        vel1_part1 = self.vel1_part1
        auxVel0_part0 = self.auxVel0_part0
        auxVel1_part0 = self.auxVel1_part0
        auxVel0_part1 = self.auxVel0_part1
        auxVel1_part1 = self.auxVel1_part1

        # |--- the values at cell-quad-points
        vel0_part0_val = vspace.value(vel0_part0, c_bcs)  # (NQ,NC)
        vel1_part0_val = vspace.value(vel1_part0, c_bcs)  # (NQ,NC)
        vel0_part1_val = vspace.value(vel0_part1, c_bcs)  # (NQ,NC)
        vel1_part1_val = vspace.value(vel1_part1, c_bcs)  # (NQ,NC)
        grad_vel0_part0_val = vspace.grad_value(vel0_part0, c_bcs)  # (NQ,NC,GD)
        grad_vel1_part0_val = vspace.grad_value(vel1_part0, c_bcs)  # (NQ,NC,GD)
        grad_vel0_part1_val = vspace.grad_value(vel0_part1, c_bcs)  # (NQ,NC,GD)
        grad_vel1_part1_val = vspace.grad_value(vel1_part1, c_bcs)  # (NQ,NC,GD)
        auxVel0_part0_val = vspace.value(auxVel0_part0, c_bcs)  # (NQ,NC)
        auxVel1_part0_val = vspace.value(auxVel1_part0, c_bcs)  # (NQ,NC)
        auxVel0_part1_val = vspace.value(auxVel0_part1, c_bcs)  # (NQ,NC)
        auxVel1_part1_val = vspace.value(auxVel1_part1, c_bcs)  # (NQ,NC)

        # |--- some coefficients
        pde = self.pde
        m = pde.m
        epsilon = pde.epsilon
        eta = pde.eta
        c_ws = self.c_ws
        cellmeasure = self.cellmeasure
        dt = self.dt
        rho_bar_n = self.rho_bar_n
        rho_bar_next = rho_bar_n  # (NQ,NC)
        nu_bar_n = self.nu_bar_n

        # |--- the (n)-th time step value
        uh_val = space.value(uh, c_bcs)  # (NQ,NC)
        vel0_val = vspace.value(self.vel0, c_bcs)
        vel1_val = vspace.value(self.vel1, c_bcs)
        grad_mu_val = self.grad_mu_val

        # # ------------------------------ # #
        # # --- to update the Xi value --- # #
        # # ------------------------------ # #
        f_val_NS = self.pde.source_NS(self.c_pp)  # (NQ,NC,GD)

        def integral_cell(X):
            # |--- X.shape: (NQ,NC)
            return np.einsum('i, ij, j->', c_ws, X, cellmeasure)  # (1,)

        # def grad_x_grad(grad_X, grad_Y):
        #     # |--- grad_X.shape: (NQ,NC,GD)
        #     GD = grad_X.shape[-1]
        #     return np.sum(grad_X * grad_Y, axis=GD)  # (NQ,NC)

        def Du_x_Du(grad_X0, grad_X1, grad_Y0, grad_Y1):
            # |--- grad_X0.shape: (NQ,NC,GD)
            Du_X = [2 * grad_X0[..., 0], grad_X0[..., 1] + grad_X1[..., 0], grad_X0[..., 1] + grad_X1[..., 0], 2 * grad_X1[..., 1]]
            Du_Y = [2 * grad_Y0[..., 0], grad_Y0[..., 1] + grad_Y1[..., 0], grad_Y0[..., 1] + grad_Y1[..., 0], 2 * grad_Y1[..., 1]]
            return Du_X[0] * Du_Y[0] + Du_X[1] * Du_Y[1] + Du_X[2] * Du_Y[2] + Du_X[3] * Du_Y[3]  # (NQ,NC)

        # |--- compute the coefficients of \Xi
        B_VC0 = (1./(2*dt) * integral_cell(rho_bar_next * (vel0_part0_val**2 + vel1_part0_val**2))
                 - 1./(2*dt) * integral_cell(rho_bar_n * (vel0_val**2 + vel1_val**2))
                 + 1./(2*dt) * integral_cell(rho_bar_n * ((auxVel0_part0_val - vel0_val)**2 + (auxVel1_part0_val - vel1_val)**2))
                 + 1./2 * integral_cell(nu_bar_n * Du_x_Du(grad_vel0_part0_val, grad_vel1_part0_val,
                                                           grad_vel0_part0_val, grad_vel1_part0_val))
                 + 1./(2*dt) * integral_cell(rho_bar_n * ((vel0_part0_val - auxVel0_part0_val)**2
                                                          + (vel1_part0_val - auxVel1_part0_val)**2))
                 - integral_cell(f_val_NS[..., 0] * auxVel0_part0_val + f_val_NS[..., 1] * auxVel1_part0_val)
                 )  # (1,)
        B_VC1 = (1./dt * integral_cell(rho_bar_next * (vel0_part0_val * vel0_part1_val + vel1_part0_val * vel1_part1_val))
                 + 1./dt * integral_cell(rho_bar_n * ((auxVel0_part0_val - vel0_val) * auxVel0_part1_val
                                                      + (auxVel1_part0_val - vel1_val) * auxVel1_part1_val))
                 + integral_cell(nu_bar_n * Du_x_Du(grad_vel0_part0_val, grad_vel1_part0_val, grad_vel0_part1_val, grad_vel1_part1_val))
                 + 1./dt * integral_cell(rho_bar_n * ((vel0_part0_val - auxVel0_part0_val) * (vel0_part1_val - auxVel0_part1_val)
                                                      + (vel1_part0_val - auxVel1_part0_val) * (vel1_part1_val - auxVel1_part1_val)))
                 - integral_cell(f_val_NS[..., 0] * auxVel0_part1_val + f_val_NS[..., 1] * auxVel1_part1_val)
                 )  # (1,)
        B_VC2 = (1./(2*dt) * integral_cell(rho_bar_next * (vel0_part1_val**2 + vel1_part1_val**2))
                 + 1./(2*dt) * integral_cell(rho_bar_n * (auxVel0_part1_val**2 + auxVel1_part1_val**2))
                 + 1./2 * integral_cell(nu_bar_n * Du_x_Du(grad_vel0_part1_val, grad_vel1_part1_val,
                                                           grad_vel0_part1_val, grad_vel1_part1_val))
                 + 1./(2*dt) * integral_cell(rho_bar_n * ((vel0_part1_val - auxVel0_part1_val)**2
                                                          + (vel1_part1_val - auxVel1_part1_val)**2))
                 )  # (1,)

        # |--- given the R_n
        R_n = self.R_n

        # |--- f_Xi = (2./dt + B_VC2) * Xi^3 + (-2./dt * R_n + B_VC1) * Xi^2 + B_VC0 * Xi
        def f_Xi(X):
            return (2. / dt + B_VC2) * X**3 + (-2. / dt * R_n + B_VC1) * X**2 + B_VC0 * X

        def diff_f_Xi(X):
            return 3 * (2. / dt + B_VC2) * X**2 + 2 * (-2. / dt * R_n + B_VC1) * X + B_VC0

        def Newton_iteration(tol, x0):
            err = 1.
            while err > tol:
                xn = x0 - 1./diff_f_Xi(x0) * f_Xi(x0)
                err = np.abs(xn - x0)
                x0 = xn
            return x0
        Xi = Newton_iteration(1.e-9, 1.)
        self.R_n = Xi
        self.Xi = Xi
        return Xi

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
        isPeriodicEdge1 = np.abs(bd_mid[:, 0] - 0.8) < 1e-8
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

