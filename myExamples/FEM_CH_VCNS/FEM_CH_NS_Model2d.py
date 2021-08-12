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


class FEM_CH_NS_Model2d:
    def __init__(self, pde, mesh, p, dt):
        self.pde = pde
        self.p = p
        self.mesh = mesh
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
        self.dt_min = pde.dt_min if hasattr(pde, 'dt_min') else self.dt
        self.s, self.alpha = self.set_CH_Coeff(dt_minimum=self.dt_min)
        self.face2dof = self.dof.face_to_dof()  # (NE,fldof)
        self.cell2dof = self.dof.cell_to_dof()  # (NC,cldof)
        self.vface2dof = self.vdof.face_to_dof()  # (NE,vfldof)
        self.vcell2dof = self.vdof.cell_to_dof()  # (NC,vcldof)
        self.NeuEdgeIdx_CH = self.set_CH_Neumann_edge()
        self.nNeu_CH = self.mesh.face_unit_normal(index=self.NeuEdgeIdx_CH)  # (NBE,2)
        self.NeuCellIdx_CH = self.mesh.ds.edge2cell[self.NeuEdgeIdx_CH, 0]
        self.NeuLocalIdx_CH = self.mesh.ds.edge2cell[self.NeuEdgeIdx_CH, 2]
        self.NeuEdgeMeasure_CH = self.mesh.entity_measure('face', index=self.NeuEdgeIdx_CH)  # (Nneu,2)
        # f_q = self.integralalg.faceintegrator
        self.f_bcs, self.f_ws = self.integralalg.faceintegrator.get_quadrature_points_and_weights()  # f_bcs.shape: (NQ,(GD-1)+1)
        self.f_pp_Neu_CH = self.mesh.bc_to_point(self.f_bcs,
                                                 index=self.NeuEdgeIdx_CH)  # f_pp.shape: (NQ,NBE,GD) the physical Gauss points
        # c_q = self.integralalg.cellintegrator
        self.c_bcs, self.c_ws = self.integralalg.faceintegrator.get_quadrature_points_and_weights()  # c_bcs.shape: (NQ,GD+1)
        self.c_pp = self.mesh.bc_to_point(self.c_bcs)  # c_pp.shape: (NQ_cell,NC,GD) the physical Gauss points
        self.phi_f = self.space.face_basis(self.f_bcs)  # (NQ,1,fldof) 实际上这里可以直接用 pspace.basis(f_bcs), 两个函数的代码是相同的
        self.phi_c = self.space.basis(self.c_bcs)  # (NQ,NC,clodf)
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

    def CH_NS_Solver_T1stOrder(self):
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

        print('    # #################################### #')
        print('      Time 1st-order scheme')
        print('    # #################################### #')

        print('    # ------------ parameters ------------ #')
        print('    s = %.4e,  alpha = %.4e,  m = %.4e,  epsilon = %.4e,  eta = %.4e' % (self.s, self.alpha, self.pde.m,
                                                                                        self.pde.epsilon, self.pde.eta))
        print('    t0 = %.4e,  T = %.4e, dt = %.4e' % (timemesh[0], timemesh[-1], dt))
        print(' ')

        phi_f_Neu_CH = space.face_basis(self.f_bcs)  # (NQ,1,fldof). 实际上这里可以直接用 pspace.basis(f_bcs), 两个函数的代码是相同的
        phi_c = space.basis(self.c_bcs)  # (NQ,NC,clodf)
        gphi_c = space.grad_basis(self.c_bcs)  # (NQ,NC,cldof,GD)

    def decoupled_CH_Solver_T1stOrder(self, uh, wh, next_t, vel0, vel1):
        """
        The docoupled solver for CH equation.
        """

        # # --- the following is
        grad_free_energy_c = self.pde.epsilon / self.pde.eta ** 2 * self.grad_free_energy_at_cells(uh, self.c_bcs)  # (NQ,NC,2)
        grad_free_energy_f = self.pde.epsilon / self.pde.eta ** 2 * \
                             self.grad_free_energy_at_faces(uh, self.f_bcs, self.NeuEdgeIdx_CH, self.NeuCellIdx_CH,
                                                            self.NeuLocalIdx_CH)  # (NQ,NE,2)
        uh_val = self.space.value(uh, self.c_bcs)  # (NQ,NC)
        guh_val_c = self.space.grad_value(uh, self.c_bcs)  # (NQ,NC,2)
        guh_val_f = self.uh_grad_value_at_faces(uh, self.f_bcs, self.NeuCellIdx_CH, self.NeuLocalIdx_CH)  # (NQ,NE,2)

        Neumann = self.pde.neumann(self.f_pp_Neu_CH, next_t, self.nNeu_CH)  # (NQ,NE)
        LaplaceNeumann = self.pde.laplace_neumann(self.f_pp_Neu_CH, next_t, self.nNeu_CH)  # (NQ,NE)
        f_val_CH = self.pde.source_CH(self.c_pp, next_t, self.pde.m, self.pde.epsilon, self.pde.eta)  # (NQ,NC)

        # # get the auxiliary equation Right-hand-side-Vector
        aux_rv = np.zeros((self.dof.number_of_global_dofs(),), dtype=self.ftype)  # (Ndof,)

        # # aux_rhs_c_0:  -1. / (epsilon * m * dt) * (uh^n,phi)_\Omega
        aux_rhs_c_0 = -1. / (self.pde.epsilon * self.pde.m) * (1 / self.dt * np.einsum('i, ij, ijk, j->jk', self.c_ws, uh_val, self.phi_c, self.cellmeasure) +
                                             np.einsum('i, ij, ijk, j->jk', self.c_ws, f_val_CH, self.phi_c, self.cellmeasure))  # (NC,cldof)
        # # aux_rhs_c_1: -s / epsilon * (\nabla uh^n, \nabla phi)_\Omega
        aux_rhs_c_1 = -self.s / self.pde.epsilon * (
                np.einsum('i, ij, ijk, j->jk', self.c_ws, guh_val_c[..., 0], self.gphi_c[..., 0], self.cellmeasure)
                + np.einsum('i, ij, ijk, j->jk', self.c_ws, guh_val_c[..., 1], self.gphi_c[..., 1], self.cellmeasure))  # (NC,cldof)
        # # aux_rhs_c_2: 1 / epsilon * (\nabla h(uh^n), \nabla phi)_\Omega
        aux_rhs_c_2 = 1. / self.pde.epsilon * (
                np.einsum('i, ij, ijk, j->jk', self.c_ws, grad_free_energy_c[..., 0], self.gphi_c[..., 0], self.cellmeasure)
                + np.einsum('i, ij, ijk, j->jk', self.c_ws, grad_free_energy_c[..., 1], self.gphi_c[..., 1], self.cellmeasure))  # (NC,cldof)

        # # aux_rhs_f_0: (\nabla wh^{n+1}\cdot n, phi)_\Gamma, wh is the solution of auxiliary equation
        aux_rhs_f_0 = self.alpha * np.einsum('i, ij, ijn, j->jn', self.f_ws, Neumann, self.phi_f, self.NeuEdgeMeasure_CH) \
                      + np.einsum('i, ij, ijn, j->jn', self.f_ws, LaplaceNeumann, self.phi_f, self.NeuEdgeMeasure_CH)  # (Nneu,fldof)
        # # aux_rhs_f_1: s / epsilon * (\nabla uh^n \cdot n, phi)_\Gamma
        aux_rhs_f_1 = self.s / self.pde.epsilon * np.einsum('i, ijk, jk, ijn, j->jn', self.f_ws, guh_val_f, self.nNeu_CH, self.phi_f,
                                              self.NeuEdgeMeasure_CH)  # (Nneu,fldof)
        # # aux_rhs_f_2: -1 / epsilon * (\nabla h(uh^n) \cdot n, phi)_\Gamma
        aux_rhs_f_2 = -1. / self.pde.epsilon * \
                      np.einsum('i, ijk, jk, ijn, j->jn', self.f_ws, grad_free_energy_f, self.nNeu_CH, self.phi_f, self.NeuEdgeMeasure_CH)  # (Nneu,fldof)

        # # --- now, we add the NS term
        vel0_val_c = self.vspace.value(vel0, self.c_bcs)  # (NQ,NC)
        vel1_val_c = self.vspace.value(vel1, self.c_bcs)  # (NQ,NC)
        vel0_val_f = self.vspace.value(vel0, self.f_bcs)[..., self.NeuEdgeIdx_CH]  # (NQ,NBE)
        vel1_val_f = self.vspace.value(vel0, self.f_bcs)[..., self.NeuEdgeIdx_CH]  # (NQ,NBE)
        vel_val_f = np.concatenate([vel0_val_f[..., np.newaxis], vel1_val_f[..., np.newaxis]], axis=2)

        aux_rhs_c_3 = -1. / (self.pde.epsilon * self.pde.m) * \
                      (np.einsum('i, ij, ijk, j->jk', self.c_ws, uh_val * vel0_val_c, self.gphi_c[..., 0], self.cellmeasure)
                       + np.einsum('i, ij, ijk, j->jk', self.c_ws, uh_val * vel1_val_c, self.gphi_c[..., 1], self.cellmeasure))  # (NC,cldof)
        aux_rhs_f_3 = 1. / (self.pde.epsilon * self.pde.m) * \
                      np.einsum('i, ijk, jk, ijn, j->jn', self.f_ws, vel_val_f, self.nNeu_CH, self.phi_f, self.NeuEdgeMeasure_CH)  # (Nneu,fldof)

        # # --- assemble the CH's aux equation
        np.add.at(aux_rv, self.cell2dof, aux_rhs_c_0 + aux_rhs_c_1 + aux_rhs_c_2)
        np.add.at(aux_rv, self.face2dof[self.NeuEdgeIdx_CH, :], aux_rhs_f_0 + aux_rhs_f_1 + aux_rhs_f_2)

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
