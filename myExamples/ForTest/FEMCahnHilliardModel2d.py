#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: FEMCahnHilliardModel2d.py
# @Author: Yongchao Zhang, Northwest University, Xi'an
# @E-mail: yoczhang@nwu.edu.cn
# @Site:
# @Time: Aug 03, 2021
# ---


__doc__ = """
The FEM Cahn-Hilliard model in 2D. 
"""

import numpy as np
from scipy.sparse import csr_matrix, spdiags, eye, bmat
from fealpy.quadrature import FEMeshIntegralAlg
from scipy.sparse.linalg import spsolve
from fealpy.boundarycondition import DirichletBC
from fealpy.decorator import timer
from fealpy.functionspace import LagrangeFiniteElementSpace


class FEMCahnHilliardModel2d:
    def __init__(self, pde, mesh, p, dt):
        self.pde = pde
        self.p = p
        self.mesh = mesh
        self.timemesh, self.dt = self.pde.time_mesh(dt)
        self.itype = self.mesh.itype
        self.ftype = self.mesh.ftype
        self.pde = pde
        self.space = LagrangeFiniteElementSpace(mesh, p)
        self.dof = self.space.dof
        self.cellmeasure = mesh.entity_measure('cell')
        self.integralalg = FEMeshIntegralAlg(self.mesh, p + 4, cellmeasure=self.cellmeasure)
        self.uh = self.space.function()
        self.wh = self.space.function()
        self.StiffMatrix = self.space.stiff_matrix()
        self.MassMatrix = self.space.mass_matrix()

    def setCoefficient_T1stOrder(self, dt_minimum=None):
        pde = self.pde
        dt_min = self.dt if dt_minimum is None else dt_minimum
        m = pde.m
        epsilon = pde.epsilon

        s = np.sqrt(4 * epsilon / (m * dt_min))
        alpha = 1. / (2 * epsilon) * (-s + np.sqrt(abs(s ** 2 - 4 * epsilon / (m * self.dt))))
        return s, alpha

    def setCoefficient_T2ndOrder(self, dt_minimum=None):
        pde = self.pde
        dt_min = self.dt if dt_minimum is None else dt_minimum
        m = pde.m
        epsilon = pde.epsilon

        s = np.sqrt(4 * (3/2) * epsilon / (m * dt_min))
        alpha = 1. / (2 * epsilon) * (-s + np.sqrt(abs(s ** 2 - 4 * (3/2) * epsilon / (m * self.dt))))
        return s, alpha

    def uh_grad_value_at_faces(self, vh, f_bcs, cellidx, localidx):
        cell2dof = self.dof.cell2dof
        f_gphi = self.space.edge_grad_basis(f_bcs, cellidx, localidx)  # (NE,NQ,cldof,GD)
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

    @timer
    def CH_Solver_T1stOrder(self):
        pde = self.pde
        timemesh = self.timemesh
        NT = len(timemesh)
        dt = self.dt
        uh = self.uh
        wh = self.wh
        sm = self.StiffMatrix
        mm = self.MassMatrix
        space = self.space
        dof = self.dof
        face2dof = dof.face_to_dof()  # (NE,fldof)
        cell2dof = dof.cell_to_dof()  # (NC,cldof)

        dt_min = pde.dt_min if hasattr(pde, 'dt_min') else dt
        s, alpha = self.setCoefficient_T1stOrder(dt_minimum=dt_min)
        m = pde.m
        epsilon = pde.epsilon
        eta = pde.eta

        print('    # #################################### #')
        print('      Time 1st-order scheme')
        print('    # #################################### #')

        print('    # ------------ parameters ------------ #')
        print('    s = %.4e,  alpha = %.4e,  m = %.4e,  epsilon = %.4e,  eta = %.4e' % (s, alpha, m, epsilon, eta))
        print('    t0 = %.4e,  T = %.4e, dt = %.4e' % (timemesh[0], timemesh[-1], dt))
        print(' ')

        idxNeuEdge = self.set_Neumann_edge()
        nBd = self.mesh.face_unit_normal(index=idxNeuEdge)  # (NBE,2)
        NeuCellIdx = self.mesh.ds.edge2cell[idxNeuEdge, 0]
        NeuLocalIdx = self.mesh.ds.edge2cell[idxNeuEdge, 2]
        neu_face_measure = self.mesh.entity_measure('face', index=idxNeuEdge)  # (Nneu,2)
        cell_measure = self.mesh.cell_area()

        f_q = self.integralalg.faceintegrator
        f_bcs, f_ws = f_q.get_quadrature_points_and_weights()  # f_bcs.shape: (NQ,(GD-1)+1)
        f_pp = self.mesh.bc_to_point(f_bcs, index=idxNeuEdge)  # f_pp.shape: (NQ,NBE,GD) the physical Gauss points
        c_q = self.integralalg.cellintegrator
        c_bcs, c_ws = c_q.get_quadrature_points_and_weights()  # c_bcs.shape: (NQ,GD+1)
        c_pp = self.mesh.bc_to_point(c_bcs)  # c_pp.shape: (NQ_cell,NC,GD) the physical Gauss points

        phi_f = space.face_basis(f_bcs)  # (NQ,1,fldof). 实际上这里可以直接用 pspace.basis(f_bcs), 两个函数的代码是相同的
        phi_c = space.basis(c_bcs)  # (NQ,NC,clodf)
        gphi_c = space.grad_basis(c_bcs)  # (NQ,NC,cldof,GD)

        # # time-looping
        print('    # ------------ begin the time-looping ------------ #')
        for nt in range(NT-1):
            currt_t = timemesh[nt]
            next_t = currt_t + dt

            if nt % max([int(NT / 10), 1]) == 0:
                print('    currt_t = %.4e' % currt_t)
            if nt == 0:
                # the initial value setting
                u_c = pde.solution(c_pp, pde.t0)  # (NQC,NC)
                gu_c = pde.gradient(c_pp, pde.t0)  # (NQC,NC,2)
                u_f = pde.solution(f_pp, pde.t0)  # (NQF,NBE)
                gu_f = pde.gradient(f_pp, pde.t0)  # (NQF,NBE,2)

                grad_free_energy_c = epsilon / eta ** 2 * (3 * np.repeat(u_c[..., np.newaxis], 2, axis=2) ** 2 * gu_c - gu_c)
                grad_free_energy_f = epsilon / eta ** 2 * (3 * np.repeat(u_f[..., np.newaxis], 2, axis=2) ** 2 * gu_f - gu_f)

                uh_val = u_c  # (NQ,NC)
                guh_val_c = gu_c  # (NQ,NC,2)
                guh_val_f = gu_f  # (NQ,NE,2)

                # gu_c_test = gu_c.copy()
                # gu_c_test[..., 0] = epsilon / eta ** 2 * (3 * u_c ** 2 * gu_c_test[..., 0] - gu_c_test[..., 0])
                # gu_c_test[..., 1] = epsilon / eta ** 2 * (3 * u_c ** 2 * gu_c_test[..., 1] - gu_c_test[..., 1])
                # gu_f_test = gu_f.copy()
                # gu_f_test[..., 0] = epsilon / eta ** 2 * (3 * u_f ** 2 * gu_f_test[..., 0] - gu_f_test[..., 0])
                # gu_f_test[..., 1] = epsilon / eta ** 2 * (3 * u_f ** 2 * gu_f_test[..., 1] - gu_f_test[..., 1])
                # print(np.allclose(gu_c_test, grad_free_energy_c))
                # print(np.allclose(gu_f_test, grad_free_energy_f))
            else:
                grad_free_energy_c = epsilon / eta ** 2 * self.grad_free_energy_at_cells(uh, c_bcs)  # (NQ,NC,2)
                grad_free_energy_f = epsilon / eta ** 2 * self.grad_free_energy_at_faces(uh, f_bcs, idxNeuEdge, NeuCellIdx,
                                                                                         NeuLocalIdx)  # (NQ,NE,2)
                uh_val = space.value(uh, c_bcs)  # (NQ,NC)
                guh_val_c = space.grad_value(uh, c_bcs)  # (NQ,NC,2)
                guh_val_f = self.uh_grad_value_at_faces(uh, f_bcs, NeuCellIdx, NeuLocalIdx)  # (NQ,NE,2)

            # # # ---------------------------------------- yc test --------------------------------------------------- # #
            # if nt == 0:
            #     def init_solution(p):
            #         return pde.solution(p, 0)
            #     uh[:] = space.interpolation(init_solution)
            # grad_free_energy_c = epsilon / eta ** 2 * self.grad_free_energy_at_cells(uh, c_bcs)  # (NQ,NC,2)
            # grad_free_energy_f = epsilon / eta ** 2 * self.grad_free_energy_at_faces(uh, f_bcs, idxNeuEdge, NeuCellIdx,
            #                                                                          NeuLocalIdx)  # (NQ,NE,2)
            # uh_val = space.value(uh, c_bcs)  # (NQ,NC)
            # guh_val_c = space.grad_value(uh, c_bcs)  # (NQ,NC,2)
            # guh_val_f = self.uh_grad_value_at_faces(uh, f_bcs, NeuCellIdx, NeuLocalIdx)  # (NQ,NE,2)
            # # # --------------------------------------- end test ---------------------------------------------------- # #

            Neumann = pde.neumann(f_pp, next_t, nBd)  # (NQ,NE)
            LaplaceNeumann = pde.laplace_neumann(f_pp, next_t, nBd)  # (NQ,NE)
            f_val = pde.source(c_pp, next_t, m, epsilon, eta)  # (NQ,NC)

            # # get the auxiliary equation Right-hand-side-Vector
            aux_rv = np.zeros((dof.number_of_global_dofs(),), dtype=self.ftype)  # (Ndof,)

            # # aux_rhs_c_0:  -1. / (epsilon * m * dt) * (uh^n,phi)_\Omega
            aux_rhs_c_0 = -1. / (epsilon * m) * (1/dt * np.einsum('i, ij, ijk, j->jk', c_ws, uh_val, phi_c, cell_measure) +
                                                 np.einsum('i, ij, ijk, j->jk', c_ws, f_val, phi_c, cell_measure))  # (NC,cldof)
            # # aux_rhs_c_1: -s / epsilon * (\nabla uh^n, \nabla phi)_\Omega
            aux_rhs_c_1 = -s / epsilon * (
                    np.einsum('i, ij, ijk, j->jk', c_ws, guh_val_c[..., 0], gphi_c[..., 0], cell_measure)
                    + np.einsum('i, ij, ijk, j->jk', c_ws, guh_val_c[..., 1], gphi_c[..., 1], cell_measure))  # (NC,cldof)
            # # aux_rhs_c_2: 1 / epsilon * (\nabla h(uh^n), \nabla phi)_\Omega
            aux_rhs_c_2 = 1. / epsilon * (
                    np.einsum('i, ij, ijk, j->jk', c_ws, grad_free_energy_c[..., 0], gphi_c[..., 0], cell_measure)
                    + np.einsum('i, ij, ijk, j->jk', c_ws, grad_free_energy_c[..., 1], gphi_c[..., 1], cell_measure))  # (NC,cldof)

            # # aux_rhs_f_0: (\nabla wh^{n+1}\cdot n, phi)_\Gamma, wh is the solution of auxiliary equation
            aux_rhs_f_0 = alpha * np.einsum('i, ij, ijn, j->jn', f_ws, Neumann, phi_f, neu_face_measure) \
                          + np.einsum('i, ij, ijn, j->jn', f_ws, LaplaceNeumann, phi_f, neu_face_measure)  # (Nneu,fldof)
            # # aux_rhs_f_1: s / epsilon * (\nabla uh^n \cdot n, phi)_\Gamma
            aux_rhs_f_1 = s / epsilon * np.einsum('i, ijk, jk, ijn, j->jn', f_ws, guh_val_f, nBd, phi_f,
                                                  neu_face_measure)  # (Nneu,fldof)
            # # aux_rhs_f_2: -1 / epsilon * (\nabla h(uh^n) \cdot n, phi)_\Gamma
            aux_rhs_f_2 = -1. / epsilon * np.einsum('i, ijk, jk, ijn, j->jn', f_ws, grad_free_energy_f, nBd, phi_f,
                                                    neu_face_measure)  # (Nneu,fldof)

            np.add.at(aux_rv, cell2dof, aux_rhs_c_0 + aux_rhs_c_1 + aux_rhs_c_2)
            np.add.at(aux_rv, face2dof[idxNeuEdge, :], aux_rhs_f_0 + aux_rhs_f_1 + aux_rhs_f_2)

            # # update the solution of auxiliary equation
            wh[:] = spsolve(sm + (alpha + s / epsilon) * mm, aux_rv)

            # # update the original solution
            orig_rv = np.zeros((dof.number_of_global_dofs(),), dtype=self.ftype)  # (Ndof,)
            wh_val = space.value(wh, c_bcs)  # (NQ,NC)
            orig_rhs_c = - np.einsum('i, ij, ijk, j->jk', c_ws, wh_val, phi_c, cell_measure)  # (NC,cldof)
            orig_rhs_f = np.einsum('i, ij, ijn, j->jn', f_ws, Neumann, phi_f, neu_face_measure)
            np.add.at(orig_rv, cell2dof, orig_rhs_c)
            np.add.at(orig_rv, face2dof[idxNeuEdge, :], orig_rhs_f)
            uh[:] = spsolve(sm - alpha * mm, orig_rv)
            # currt_l2err = self.currt_error(uh, next_t)
            print('max(uh) = ', max(uh))
            # if max(uh[:]) > 1e5:
            #     break
        print('    # ------------ end the time-looping ------------ #\n')

        l2err, h1err = self.currt_error(uh, timemesh[-1])
        print('    # ------------ the last errors ------------ #')
        print('    l2err = %.4e, h1err = %.4e' % (l2err, h1err))
        return l2err, h1err

    def CH_Solver_T2ndOrder(self):
        pde = self.pde
        timemesh = self.timemesh
        NT = len(timemesh)
        space = self.space
        dt = self.dt
        uh = self.uh
        last_uh = space.function()
        wh = self.wh
        sm = self.StiffMatrix
        mm = self.MassMatrix
        dof = self.dof
        face2dof = dof.face_to_dof()  # (NE,fldof)
        cell2dof = dof.cell_to_dof()  # (NC,cldof)

        dt_min = pde.dt_min if hasattr(pde, 'dt_min') else dt
        s, alpha = self.setCoefficient_T2ndOrder(dt_minimum=dt_min)
        m = pde.m
        epsilon = pde.epsilon
        eta = pde.eta

        print('    # #################################### #')
        print('      Time 2nd-order scheme')
        print('    # #################################### #')

        print('    # ------------ parameters ------------ #')
        print('    s = %.4e,  alpha = %.4e,  m = %.4e,  epsilon = %.4e,  eta = %.4e' % (s, alpha, m, epsilon, eta))
        print('    t0 = %.4e,  T = %.4e, dt = %.4e' % (timemesh[0], timemesh[-1], dt))
        print(' ')

        idxNeuEdge = self.set_Neumann_edge()
        nBd = self.mesh.face_unit_normal(index=idxNeuEdge)  # (NBE,2)
        NeuCellIdx = self.mesh.ds.edge2cell[idxNeuEdge, 0]
        NeuLocalIdx = self.mesh.ds.edge2cell[idxNeuEdge, 2]
        neu_face_measure = self.mesh.entity_measure('face', index=idxNeuEdge)  # (Nneu,2)
        cell_measure = self.mesh.cell_area()

        f_q = self.integralalg.faceintegrator
        f_bcs, f_ws = f_q.get_quadrature_points_and_weights()  # f_bcs.shape: (NQ,(GD-1)+1)
        f_pp = self.mesh.bc_to_point(f_bcs, index=idxNeuEdge)  # f_pp.shape: (NQ,NBE,GD) the physical Gauss points
        c_q = self.integralalg.cellintegrator
        c_bcs, c_ws = c_q.get_quadrature_points_and_weights()  # c_bcs.shape: (NQ,GD+1)
        c_pp = self.mesh.bc_to_point(c_bcs)  # c_pp.shape: (NQ_cell,NC,GD) the physical Gauss points

        phi_f = space.face_basis(f_bcs)  # (NQ,1,fldof). 实际上这里可以直接用 pspace.basis(f_bcs), 两个函数的代码是相同的
        phi_c = space.basis(c_bcs)  # (NQ,NC,clodf)
        gphi_c = space.grad_basis(c_bcs)  # (NQ,NC,cldof,GD)

        # # time-looping
        print('    # ------------ begin the time-looping ------------ #')

        def init_solution(p):
            return pde.solution(p, 0)
        last_uh[:] = space.interpolation(init_solution)
        for nt in range(NT-1):
            currt_t = timemesh[nt]
            next_t = currt_t + dt
            if nt % max([int(NT/10), 1]) == 0:
                print('    currt_t = %.4e' % currt_t)
            if nt == 0:
                # the initial value setting
                s, alpha = self.setCoefficient_T1stOrder(dt_minimum=dt_min)

                u_c = pde.solution(c_pp, pde.t0)  # (NQC,NC)
                gu_c = pde.gradient(c_pp, pde.t0)  # (NQC,NC,2)
                u_f = pde.solution(f_pp, pde.t0)  # (NQF,NBE)
                gu_f = pde.gradient(f_pp, pde.t0)  # (NQF,NBE,2)

                grad_free_energy_c = epsilon / eta ** 2 * (3 * np.repeat(u_c[..., np.newaxis], 2, axis=2) ** 2 * gu_c - gu_c)
                grad_free_energy_f = epsilon / eta ** 2 * (3 * np.repeat(u_f[..., np.newaxis], 2, axis=2) ** 2 * gu_f - gu_f)

                uh_val = u_c  # (NQ,NC)
                guh_val_c = gu_c  # (NQ,NC,2)
                guh_val_f = gu_f  # (NQ,NE,2)
            else:
                s, alpha = self.setCoefficient_T2ndOrder(dt_minimum=dt_min)
                grad_free_energy_c = epsilon / eta ** 2 * self.grad_free_energy_at_cells(2*uh - last_uh, c_bcs)  # (NQ,NC,2)
                grad_free_energy_f = epsilon / eta ** 2 * self.grad_free_energy_at_faces(2*uh - last_uh, f_bcs, idxNeuEdge, NeuCellIdx, NeuLocalIdx)  # (NQ,NE,2)
                uh_val = space.value(2 * uh - 1/2 * last_uh, c_bcs)  # (NQ,NC)
                guh_val_c = space.grad_value(2*uh - last_uh, c_bcs)  # (NQ,NC,2)
                guh_val_f = self.uh_grad_value_at_faces(2*uh - last_uh, f_bcs, NeuCellIdx, NeuLocalIdx)  # (NQ,NE,2)

                last_uh[:] = uh[:]  # update the last_uh

            Neumann = pde.neumann(f_pp, next_t, nBd)  # (NQ,NE)
            LaplaceNeumann = pde.laplace_neumann(f_pp, next_t, nBd)  # (NQ,NE)
            f_val = pde.source(c_pp, next_t, m, epsilon, eta)  # (NQ,NC)

            # # get the auxiliary equation Right-hand-side-Vector
            aux_rv = np.zeros((dof.number_of_global_dofs(),), dtype=self.ftype)  # (Ndof,)
            aux_rv_temp = np.zeros((dof.number_of_global_dofs(),), dtype=self.ftype)

            # # aux_rhs_c_0:  -1. / (epsilon * m * dt) * (uh^n,phi)_\Omega
            aux_rhs_c_0 = -1. / (epsilon * m) * (1/dt * np.einsum('i, ij, ijk, j->jk', c_ws, uh_val, phi_c, cell_measure) +
                                                 np.einsum('i, ij, ijk, j->jk', c_ws, f_val, phi_c, cell_measure))  # (NC,cldof)
            # # aux_rhs_c_1: -s / epsilon * (\nabla uh^n, \nabla phi)_\Omega
            aux_rhs_c_1 = -s / epsilon * (
                    np.einsum('i, ij, ijk, j->jk', c_ws, guh_val_c[..., 0], gphi_c[..., 0], cell_measure)
                    + np.einsum('i, ij, ijk, j->jk', c_ws, guh_val_c[..., 1], gphi_c[..., 1], cell_measure))  # (NC,cldof)
            # # aux_rhs_c_2: 1 / epsilon * (\nabla h(uh^n), \nabla phi)_\Omega
            aux_rhs_c_2 = 1. / epsilon * (
                    np.einsum('i, ij, ijk, j->jk', c_ws, grad_free_energy_c[..., 0], gphi_c[..., 0], cell_measure)
                    + np.einsum('i, ij, ijk, j->jk', c_ws, grad_free_energy_c[..., 1], gphi_c[..., 1], cell_measure))  # (NC,cldof)

            # # aux_rhs_f_0: (\nabla wh^{n+1}\cdot n, phi)_\Gamma, wh is the solution of auxiliary equation
            aux_rhs_f_0 = alpha * np.einsum('i, ij, ijn, j->jn', f_ws, Neumann, phi_f, neu_face_measure) \
                          + np.einsum('i, ij, ijn, j->jn', f_ws, LaplaceNeumann, phi_f, neu_face_measure)  # (Nneu,fldof)
            # # aux_rhs_f_1: s / epsilon * (\nabla uh^n \cdot n, phi)_\Gamma
            aux_rhs_f_1 = s / epsilon * np.einsum('i, ijk, jk, ijn, j->jn', f_ws, guh_val_f, nBd, phi_f,
                                                  neu_face_measure)  # (Nneu,fldof)
            # # aux_rhs_f_2: -1 / epsilon * (\nabla h(uh^n) \cdot n, phi)_\Gamma
            aux_rhs_f_2 = -1. / epsilon * np.einsum('i, ijk, jk, ijn, j->jn', f_ws, grad_free_energy_f, nBd, phi_f,
                                                    neu_face_measure)  # (Nneu,fldof)

            np.add.at(aux_rv_temp, cell2dof, aux_rhs_c_0)
            np.add.at(aux_rv, cell2dof, aux_rhs_c_0 + aux_rhs_c_1 + aux_rhs_c_2)
            np.add.at(aux_rv, face2dof[idxNeuEdge, :], aux_rhs_f_0 + aux_rhs_f_1 + aux_rhs_f_2)

            # # update the solution of auxiliary equation
            wh[:] = spsolve(sm + (alpha + s / epsilon) * mm, aux_rv)

            # # update the original solution
            orig_rv = np.zeros((dof.number_of_global_dofs(),), dtype=self.ftype)  # (Ndof,)
            wh_val = space.value(wh, c_bcs)  # (NQ,NC)
            orig_rhs_c = - np.einsum('i, ij, ijk, j->jk', c_ws, wh_val, phi_c, cell_measure)  # (NC,cldof)
            orig_rhs_f = np.einsum('i, ij, ijn, j->jn', f_ws, Neumann, phi_f, neu_face_measure)
            np.add.at(orig_rv, cell2dof, orig_rhs_c)
            np.add.at(orig_rv, face2dof[idxNeuEdge, :], orig_rhs_f)
            uh[:] = spsolve(sm - alpha * mm, orig_rv)
        print('    # ------------ end the time-looping ------------ #\n')

        l2err, h1err = self.currt_error(uh, timemesh[-1])
        print('    # ------------ the last errors ------------ #')
        print('    l2err = %.4e, h1err = %.4e' % (l2err, h1err))
        return l2err, h1err

    def currt_error(self, uh, t):
        pde = self.pde

        def currt_solution(p):
            return pde.solution(p, t)
        l2err = self.space.integralalg.L2_error(currt_solution, uh)

        def currt_grad_solution(p):
            return pde.gradient(p, t)
        h1err = self.space.integralalg.L2_error(currt_grad_solution, uh.grad_value)

        return l2err, h1err

    def set_Dirichlet_edge(self, idxDirEdge=None):
        if idxDirEdge is not None:
            return idxDirEdge

        mesh = self.mesh
        edge2cell = mesh.ds.edge_to_cell()
        isBdEdge = (edge2cell[:, 0] == edge2cell[:, 1])  # (NE,), the bool vars, to get the boundary edges

        isDirEdge = isBdEdge  # here, we set all the boundary edges are Dir edges
        idxDirEdge, = np.nonzero(isDirEdge)  # (NE_Dir,)

        return idxDirEdge

    def set_Neumann_edge(self, idxNeuEdge=None):
        if idxNeuEdge is not None:
            return idxNeuEdge

        mesh = self.mesh
        edge2cell = mesh.ds.edge_to_cell()
        bdEdge = (edge2cell[:, 0] == edge2cell[:, 1])  # the bool vars, to get the boundary edges

        isNeuEdge = bdEdge  # here, we first set all the boundary edges are Neu edges
        idxNeuEdge, = np.nonzero(isNeuEdge)  # (NE_Dir,)
        return idxNeuEdge
