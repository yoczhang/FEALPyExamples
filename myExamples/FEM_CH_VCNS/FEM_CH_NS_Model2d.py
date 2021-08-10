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

    def set_CH_Coeff(self, dt_minimum=None):
        pde = self.pde
        dt_min = self.dt if dt_minimum is None else dt_minimum
        m = pde.m
        epsilon = pde.epsilon

        if pde.timeScheme in {1, '1', '1st', '1st-order', 'first', 'first-order', 'firstorder'}:
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

        idxNeuEdge_CH = self.set_CH_Neumann_edge()
        n_Bd_CH = self.mesh.face_unit_normal(index=idxNeuEdge_CH)  # (NBE,2)
        NeuCellIdx_CH = self.mesh.ds.edge2cell[idxNeuEdge_CH, 0]
        NeuLocalIdx_CH = self.mesh.ds.edge2cell[idxNeuEdge_CH, 2]
        Neu_face_measure_CH = self.mesh.entity_measure('face', index=idxNeuEdge_CH)  # (Nneu,2)
        cell_measure = self.mesh.cell_area()

        f_q = self.integralalg.faceintegrator
        f_bcs, f_ws = f_q.get_quadrature_points_and_weights()  # f_bcs.shape: (NQ,(GD-1)+1)
        f_pp_Neu_CH = self.mesh.bc_to_point(f_bcs, index=idxNeuEdge_CH)  # f_pp.shape: (NQ,NBE,GD) the physical Gauss points
        c_q = self.integralalg.cellintegrator
        c_bcs, c_ws = c_q.get_quadrature_points_and_weights()  # c_bcs.shape: (NQ,GD+1)
        c_pp = self.mesh.bc_to_point(c_bcs)  # c_pp.shape: (NQ_cell,NC,GD) the physical Gauss points

        phi_f_Neu_CH = space.face_basis(f_bcs)  # (NQ,1,fldof). 实际上这里可以直接用 pspace.basis(f_bcs), 两个函数的代码是相同的
        phi_c = space.basis(c_bcs)  # (NQ,NC,clodf)
        gphi_c = space.grad_basis(c_bcs)  # (NQ,NC,cldof,GD)


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










