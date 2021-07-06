#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: FEMNavierStokesModel2d.py
# @Author: Yongchao Zhang, Northwest University, Xi'an
# @E-mail: yoczhang@nwu.edu.cn
# @Site:
# @Time: Jul 02, 2021
# ---


__doc__ = """
The FEM Navier-Stokes model in 2D. 
"""

import numpy as np
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.quadrature import FEMeshIntegralAlg


class FEMNavierStokesModel2d:
    def __init__(self, pde, mesh, p, dt, T):
        self.p = p
        self.mesh = mesh
        self.dt = dt
        self.T = T
        self.itype = self.mesh.itype
        self.ftype = self.mesh.ftype
        self.pde = pde
        self.vspace = LagrangeFiniteElementSpace(mesh, p+1)
        self.pspace = LagrangeFiniteElementSpace(mesh, p)
        self.vdof = self.vspace.dof
        self.pdof = self.pspace.dof
        self.cellmeasure = mesh.entity_measure('cell')
        self.integralalg = FEMeshIntegralAlg(self.mesh, p+4, cellmeasure=self.cellmeasure)
        self.uh0 = self.vspace.function()
        self.uh1 = self.vspace.function()
        self.ph = self.pspace.function()

    def NS_VC_Solver(self):
        """
        The Navier-Stokes Velocity-Correction scheme solver.
        """
        pde = self.pde
        dt = self.dt
        uh0 = self.uh0
        uh1 = self.uh1
        ph = self.ph
        vspace = self.vspace
        pspace = self.pspace
        vdof = self.vdof
        pdof = self.pdof
        pface2dof = pdof.face_to_dof()
        pcell2dof = pdof.cell_to_dof()
        # n = self.mesh.face_unit_normal()

        idxDirEdge = self.set_Dirichlet_edge()

        pDirDof = pface2dof[idxDirEdge]
        n_Dir = self.mesh.face_unit_normal(index=idxDirEdge)  # (NDir,2)
        Dir_measure = self.mesh.entity_measure('face', index=idxDirEdge)  # (NDir,2)


        # vgdof = self.vspace.number_of_global_dofs()
        # pgdof = self.pspace.number_of_global_dofs()
        # init_uh0

        f_q = self.integralalg.faceintegrator
        f_bcs, f_ws = f_q.get_quadrature_points_and_weights()  # f_bcs.shape: (NQ,(GD-1)+1)
        f_pp = self.mesh.bc_to_point(f_bcs, index=idxDirEdge)  # f_pp.shape: (NQ,NDir,GD) the physical Gauss points
        c_q = self.integralalg.cellintegrator
        c_bcs, c_ws = c_q.get_quadrature_points_and_weights()  # c_bcs.shape: (NQ,GD+1)
        c_pp = self.mesh.bc_to_point(c_bcs)  # c_pp.shape: (NQ_cell,NC,GD) the physical Gauss points

        last_uh0 = vspace.function()
        last_uh1 = vspace.function()

        # # t^{n+1}: Pressure-Left-StiffMatrix
        plsm = self.pspace.stiff_matrix()

        # # t^{n+1}: Velocity-Left-MassMatrix and -StiffMatrix
        ulmm = self.vspace.mass_matrix()
        ulsm = self.vspace.stiff_matrix()

        for nt in range(int(self.T/dt)):
            curr_t = nt * dt
            next_t = curr_t + dt

            # ---------------------------------------
            # 1st-step: get the p^{n+1}
            # ---------------------------------------
            # # Pressure-Right-Matrix
            # compute cell integration
            # 1. (uh^n/dt, \nabla q)
            if curr_t == 0.:
                # for Dirichlet-face-integration
                last_gu_val0 = self.pde.grad_velocity0(f_pp, 0)  # grad_u0: (NQ,NDir,GD)
                last_gu_val1 = self.pde.grad_velocity1(f_pp, 0)  # grad_u1: (NQ,NDir,GD)

                # for cell-integration
                last_u_val = self.pde.velocityInitialValue(c_pp)  # (NQ,NC,GD)
                last_u_val0 = last_u_val[..., 0]  # (NQ,NC)
                last_u_val1 = last_u_val[..., 1]  # (NQ,NC)

                last_nolinear_val0 = self.pde.NS_nolinearTerm_0(c_pp, 0)  # (NQ,NC)
                last_nolinear_val1 = self.pde.NS_nolinearTerm_1(c_pp, 0)  # (NQ,NC)
            else:
                # for Dirichlet-face-integration
                last_gu_val0 = vspace.grad_value(last_uh0, f_bcs)  # grad_u0: (NQ,NDir,GD)
                last_gu_val1 = vspace.grad_value(last_uh1, f_bcs)  # grad_u1: (NQ,NDir,GD)

                # for cell-integration
                last_u_val0 = vspace.value(last_uh0, c_bcs)
                last_u_val1 = vspace.value(last_uh0, c_bcs)

                last_nolinear_val = self.NSNolinearTerm(last_uh0, last_uh1, c_bcs)  # last_nolinear_val.shape: (NQ,NC,GD)
                last_nolinear_val0 = last_nolinear_val[..., 0]  # (NQ,NC)
                last_nolinear_val1 = last_nolinear_val[..., 1]  # (NQ,NC)

            uDir_val = self.pde.dirichlet(f_pp, next_t)  # (NQ,NDir,GD)
            f_val = self.pde.source(c_pp, next_t)  # (NQ,NC,GD)

            p_phi = pspace.face_basis(f_bcs)  # (NQ,1,ldof). 实际上这里可以直接用 pspace.basis(f_bcs), 两个函数的代码是相同的
            p_gphi_f = pspace.grad_basis(f_bcs)  # (NQ_face,NDir,lodf,GD)
            p_gphi_c = pspace.grad_basis(c_bcs)  # (NQ_cell,NC,ldof,GD)

            # for Dirichlet faces integration
            dir_int0 = -1/dt * np.einsum('ijk, jk, imn, j->jn', uDir_val, n_Dir, p_phi, Dir_measure)  # (NDir,ldof)
            dir_int1 = - self.pde.nu * np.einsum('j, ijk, ', n_Dir[:, 1], last_gu_val1[..., 0]-last_gu_val0[..., 1], p_gphi_f[..., 0], Dir_measure)







    def NSNolinearTerm(self, uh0, uh1, bcs):
        vspace = self.vspace
        val0 = vspace.value(uh0, bcs)  # val0.shape: (NQ,NC)
        val1 = vspace.value(uh1, bcs)  # val1.shape: (NQ,NC)
        gval0 = vspace.grad_value(uh0, bcs)  # guh0.shape: (NQ,NC,2)
        gval1 = vspace.grad_value(uh1, bcs)

        NSNolinear = np.empty(gval0.shape, dtype=self.ftype)  # NSNolinear.shape: (NQ,NC,2)

        NSNolinear[..., 0] = val0 * gval0[..., 0] + val1 * gval0[..., 1]
        NSNolinear[..., 1] = val0 * gval1[..., 0] + val1 * gval1[..., 1]

        return NSNolinear

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

        issetNeuEdge = 'no'
        if issetNeuEdge == 'no':
            isNeuEdge = None

        idxNeuEdge, = np.nonzero(isNeuEdge)  # (NE_Dir,)

        return idxNeuEdge























