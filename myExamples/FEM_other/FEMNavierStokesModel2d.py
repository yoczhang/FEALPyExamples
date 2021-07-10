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
from scipy.sparse import csr_matrix, spdiags, eye, bmat
from fealpy.quadrature import FEMeshIntegralAlg
from scipy.sparse.linalg import spsolve
from fealpy.functionspace import LagrangeFiniteElementSpace
# from LagrangeFiniteElemenSpace_mine import LagrangeFiniteElementSpace


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
        pface2dof = pdof.face_to_dof()  # (NE,fldof)
        pcell2dof = pdof.cell_to_dof()  # (NC,cldof)
        # n = self.mesh.face_unit_normal()

        idxDirEdge = self.set_Dirichlet_edge()
        cellidxDir = self.mesh.ds.edge2cell[idxDirEdge, 0]
        localidxDir = self.mesh.ds.edge2cell[idxDirEdge, 2]

        Dir_face2dof = pface2dof[idxDirEdge, :]  # (NDir,flodf)
        Dir_cell2dof = pcell2dof[cellidxDir, :]  # (NDir,cldof)
        n_Dir = self.mesh.face_unit_normal(index=idxDirEdge)  # (NDir,2)
        dir_face_measure = self.mesh.entity_measure('face', index=idxDirEdge)  # (NDir,2)
        cell_measure = self.mesh.cell_area()
        dir_cell_measure = self.mesh.cell_area(cellidxDir)

        f_q = self.integralalg.faceintegrator
        f_bcs, f_ws = f_q.get_quadrature_points_and_weights()  # f_bcs.shape: (NQ,(GD-1)+1)
        f_pp = self.mesh.bc_to_point(f_bcs, index=idxDirEdge)  # f_pp.shape: (NQ,NDir,GD) the physical Gauss points

        c_q = self.integralalg.cellintegrator
        c_bcs, c_ws = c_q.get_quadrature_points_and_weights()  # c_bcs.shape: (NQ,GD+1)
        c_pp = self.mesh.bc_to_point(c_bcs)  # c_pp.shape: (NQ_cell,NC,GD) the physical Gauss points

        last_uh0 = vspace.function()
        last_uh1 = vspace.function()
        next_ph = pspace.function()

        # # t^{n+1}: Pressure-Left-StiffMatrix
        plsm = self.pspace.stiff_matrix()
        basis_int = pspace.integral_basis()
        p_phi = pspace.face_basis(f_bcs)  # (NQ,1,ldof). 实际上这里可以直接用 pspace.basis(f_bcs), 两个函数的代码是相同的
        p_gphi_f = pspace.edge_grad_basis(f_bcs, cellidxDir, localidxDir)  # (NDir,NQ,cldof,GD)
        p_gphi_c = pspace.grad_basis(c_bcs)  # (NQ_cell,NC,ldof,GD)

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
            if curr_t == 0.:
                # for Dirichlet-face-integration
                last_gu_val0 = pde.grad_velocity0(f_pp, 0)  # grad_u0: (NQ,NDir,GD)
                last_gu_val1 = pde.grad_velocity1(f_pp, 0)  # grad_u1: (NQ,NDir,GD)

                # for cell-integration
                last_u_val = pde.velocityInitialValue(c_pp)  # (NQ,NC,GD)
                last_u_val0 = last_u_val[..., 0]  # (NQ,NC)
                last_u_val1 = last_u_val[..., 1]  # (NQ,NC)

                last_nolinear_val0 = pde.NS_nolinearTerm_0(c_pp, 0)  # (NQ,NC)
                last_nolinear_val1 = pde.NS_nolinearTerm_1(c_pp, 0)  # (NQ,NC)
            else:
                # for Dirichlet-face-integration
                last_gu_val0 = vspace.grad_value(last_uh0, f_bcs)  # grad_u0: (NQ,NDir,GD)
                last_gu_val1 = vspace.grad_value(last_uh1, f_bcs)  # grad_u1: (NQ,NDir,GD)

                # for cell-integration
                last_u_val0 = vspace.value(last_uh0, c_bcs)
                last_u_val1 = vspace.value(last_uh1, c_bcs)

                last_nolinear_val = self.NSNolinearTerm(last_uh0, last_uh1, c_bcs)  # last_nolinear_val.shape: (NQ,NC,GD)
                last_nolinear_val0 = last_nolinear_val[..., 0]  # (NQ,NC)
                last_nolinear_val1 = last_nolinear_val[..., 1]  # (NQ,NC)

            uDir_val = pde.dirichlet(f_pp, next_t)  # (NQ,NDir,GD)
            f_val = pde.source(c_pp, next_t)  # (NQ,NC,GD)

            # # to get the right-hand vector
            prv = np.zeros((pdof.number_of_global_dofs(),), dtype=self.ftype)  # (Npdof,)

            # for Dirichlet faces integration
            dir_int0 = -1/dt * np.einsum('i, ijk, jk, ijn, j->jn', f_ws, uDir_val, n_Dir, p_phi, dir_face_measure)  # (NDir,fldof)
            dir_int1 = - pde.nu * (np.einsum('i, j, ij, jin, j->jn', f_ws, n_Dir[:, 1], last_gu_val1[..., 0]-last_gu_val0[..., 1],
                                             p_gphi_f[..., 0], dir_face_measure)
                                   + np.einsum('i, j, ij, jin, j->jn', f_ws, -n_Dir[:, 0], last_gu_val1[..., 0]-last_gu_val0[..., 1],
                                               p_gphi_f[..., 1], dir_face_measure))  # (NDir,cldof)
            # for cell integration
            cell_int0 = 1/dt * (np.einsum('i, ij, ijk, j->jk', c_ws, last_u_val0, p_gphi_c[..., 0], cell_measure)
                                + np.einsum('i, ij, ijk, j->jk', c_ws, last_u_val1, p_gphi_c[..., 1], cell_measure))  # (NC,cldof)
            cell_int1 = -(np.einsum('i, ij, ijk, j->jk', c_ws, last_nolinear_val0, p_gphi_c[..., 0], cell_measure)
                          + np.einsum('i, ij, ijk, j->jk', c_ws, last_nolinear_val1, p_gphi_c[..., 1], cell_measure))  # (NC,cldof)
            cell_int2 = (np.einsum('i, ij, ijk, j->jk', c_ws, f_val[..., 0], p_gphi_c[..., 0], cell_measure)
                         + np.einsum('i, ij, ijk, j->jk', c_ws, f_val[..., 1], p_gphi_c[..., 1], cell_measure))  # (NC,cldof)

            np.add.at(prv, Dir_face2dof, dir_int0)
            np.add.at(prv, Dir_cell2dof, dir_int1)
            np.add.at(prv, pcell2dof, cell_int0 + cell_int1 + cell_int2)

            # pressure satisfies \int_\Omega p = 0
            plsm = bmat([[plsm, basis_int.reshape(-1, 1)], [basis_int, None]], format='csr')
            prv = np.r_[prv, 0]
            next_ph[:] = spsolve(plsm, prv)[:-1]  # we have added one addtional dof

            pl2err = pspace.integralalg.L2_error(pde.scalar_zero_fun, next_ph)

            print('end of func')

            # update the



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























