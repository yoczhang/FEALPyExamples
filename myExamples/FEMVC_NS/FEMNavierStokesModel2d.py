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
from fealpy.boundarycondition import DirichletBC
from fealpy.decorator import timer
from fealpy.functionspace import LagrangeFiniteElementSpace
# from LagrangeFiniteElemenSpace_mine import LagrangeFiniteElementSpace


class FEMNavierStokesModel2d:
    def __init__(self, pde, mesh, p, dt):
        self.p = p
        self.mesh = mesh
        self.pde = pde
        self.timemesh, self.dt = self.pde.time_mesh(dt)
        self.itype = self.mesh.itype
        self.ftype = self.mesh.ftype
        self.vspace = LagrangeFiniteElementSpace(mesh, p+1)
        self.pspace = LagrangeFiniteElementSpace(mesh, p)
        self.vdof = self.vspace.dof
        self.pdof = self.pspace.dof
        self.cellmeasure = mesh.entity_measure('cell')
        self.integralalg = FEMeshIntegralAlg(self.mesh, p+4, cellmeasure=self.cellmeasure)
        self.uh0 = self.vspace.function()
        self.uh1 = self.vspace.function()
        self.ph = self.pspace.function()

    @timer
    def NS_VC_Solver_T1stOrder(self):
        """
        The Navier-Stokes Velocity-Correction scheme solver.
        """
        pde = self.pde
        dt = self.dt
        timemesh = self.timemesh
        uh0 = self.uh0
        uh1 = self.uh1
        ph = self.ph
        vspace = self.vspace
        pspace = self.pspace
        vdof = self.vdof
        pdof = self.pdof
        pface2dof = pdof.face_to_dof()  # (NE,fldof)
        pcell2dof = pdof.cell_to_dof()  # (NC,cldof)
        ucell2dof = vspace.cell_to_dof()  # (NC,cldof)
        vface2dof = vdof.face_to_dof()  # (NE,fldof)

        idxDirEdge = self.set_Dirichlet_edge()
        cellidxDir = self.mesh.ds.edge2cell[idxDirEdge, 0]
        localidxDir = self.mesh.ds.edge2cell[idxDirEdge, 2]

        Dir_face2dof = pface2dof[idxDirEdge, :]  # (NDir,flodf)
        Dir_cell2dof = pcell2dof[cellidxDir, :]  # (NDir,cldof)
        n_Dir = self.mesh.face_unit_normal(index=idxDirEdge)  # (NDir,2)
        dir_face_measure = self.mesh.entity_measure('face', index=idxDirEdge)  # (NDir,2)
        cell_measure = self.mesh.cell_area()
        dir_cell_measure = self.mesh.cell_area(cellidxDir)

        idxNeuEdge = self.mesh.ds.boundary_face_index()  # here Neumann edges are all the boundary
        n_Neu = self.mesh.face_unit_normal(index=idxNeuEdge)  # (NNeu,2)
        neu_face_measure = self.mesh.entity_measure('face', index=idxNeuEdge)  # (NNeu,2)
        Neu_face2dof = vface2dof[idxNeuEdge, :]  # (NNeu,flodf)

        f_q = self.integralalg.faceintegrator
        f_bcs, f_ws = f_q.get_quadrature_points_and_weights()  # f_bcs.shape: (NQ,(GD-1)+1)
        f_pp = self.mesh.bc_to_point(f_bcs, index=idxDirEdge)  # f_pp.shape: (NQ,NDir,GD) the physical Gauss points
        f_pp_Neu = self.mesh.bc_to_point(f_bcs, index=idxNeuEdge)  # f_pp.shape: (NQ,NNeu,GD) the physical Gauss points

        c_q = self.integralalg.cellintegrator
        c_bcs, c_ws = c_q.get_quadrature_points_and_weights()  # c_bcs.shape: (NQ,GD+1)
        c_pp = self.mesh.bc_to_point(c_bcs)  # c_pp.shape: (NQ_cell,NC,GD) the physical Gauss points

        # # t^{n+1}: Pressure-Left-StiffMatrix
        plsm = self.pspace.stiff_matrix()
        basis_int = pspace.integral_basis()
        p_phi = pspace.face_basis(f_bcs)  # (NQ,1,fldof). 实际上这里可以直接用 pspace.basis(f_bcs), 两个函数的代码是相同的
        p_gphi_f = pspace.edge_grad_basis(f_bcs, cellidxDir, localidxDir)  # (NDir,NQ,cldof,GD)
        p_gphi_c = pspace.grad_basis(c_bcs)  # (NQ_cell,NC,ldof,GD)

        # # t^{n+1}: Velocity-Left-MassMatrix and -StiffMatrix
        ulmm = self.vspace.mass_matrix()
        ulsm = self.vspace.stiff_matrix()
        u_phi = vspace.basis(c_bcs)  # (NQ,1,cldof)
        u_phi_f = vspace.face_basis(f_bcs)  # (NQ,1,fldof) 实际上这里可以直接用 vspace.basis(f_bcs), 两个函数的代码是相同的

        NT = len(timemesh)
        for nt in range(NT-1):
            currt_t = timemesh[nt]
            next_t = currt_t + dt

            # ---------------------------------------
            # 1st-step: get the p^{n+1}
            # ---------------------------------------
            # # Pressure-Right-Matrix
            if currt_t == 0.:
                # for Dirichlet-face-integration
                gu_val0 = pde.grad_velocity0(f_pp, 0)  # grad_u0: (NQ,NDir,GD)
                gu_val1 = pde.grad_velocity1(f_pp, 0)  # grad_u1: (NQ,NDir,GD)

                # for cell-integration
                u_val = pde.velocityInitialValue(c_pp)  # (NQ,NC,GD)
                u_val0 = u_val[..., 0]  # (NQ,NC)
                u_val1 = u_val[..., 1]  # (NQ,NC)

                nolinear_val0 = pde.NS_nolinearTerm_0(c_pp, 0)  # (NQ,NC)
                nolinear_val1 = pde.NS_nolinearTerm_1(c_pp, 0)  # (NQ,NC)
            else:
                # for Dirichlet-face-integration
                gu_val0 = self.uh_grad_value_at_faces(uh0, f_bcs, cellidxDir, localidxDir)  # grad_u0: (NQ,NDir,GD)
                gu_val1 = self.uh_grad_value_at_faces(uh1, f_bcs, cellidxDir, localidxDir)  # grad_u0: (NQ,NDir,GD)

                # for cell-integration
                u_val0 = vspace.value(uh0, c_bcs)  # (NQ,NC)
                u_val1 = vspace.value(uh1, c_bcs)

                nolinear_val = self.NSNolinearTerm(uh0, uh1, c_bcs)  # last_nolinear_val.shape: (NQ,NC,GD)
                nolinear_val0 = nolinear_val[..., 0]  # (NQ,NC)
                nolinear_val1 = nolinear_val[..., 1]  # (NQ,NC)

            Neumann_0 = pde.neumann_0(f_pp_Neu, next_t, n_Neu)  # (NQ,NE)
            Neumann_1 = pde.neumann_1(f_pp_Neu, next_t, n_Neu)  # (NQ,NE)
            uDir_val = pde.dirichlet(f_pp, next_t)  # (NQ,NDir,GD)
            f_val = pde.source(c_pp, next_t)  # (NQ,NC,GD)
            curl_uh_dir = gu_val1[..., 0]-gu_val0[..., 1]  # (NQ,NDir)

            # # --- to update the pressure value --- # #
            # # to get the Pressure's Right-hand Vector
            prv = np.zeros((pdof.number_of_global_dofs(),), dtype=self.ftype)  # (Npdof,)

            # for Dirichlet faces integration
            dir_int0 = -1/dt * np.einsum('i, ijk, jk, ijn, j->jn', f_ws, uDir_val, n_Dir, p_phi, dir_face_measure)  # (NDir,fldof)
            dir_int1 = - pde.nu * ( np.einsum('i, j, ij, jin, j->jn', f_ws, n_Dir[:, 1], curl_uh_dir,
                                              p_gphi_f[..., 0], dir_face_measure)
                                    + np.einsum('i, j, ij, jin, j->jn', f_ws, -n_Dir[:, 0], curl_uh_dir,
                                                p_gphi_f[..., 1], dir_face_measure))  # (NDir,cldof)
            # for cell integration
            cell_int0 = 1/dt * (np.einsum('i, ij, ijk, j->jk', c_ws, u_val0, p_gphi_c[..., 0], cell_measure)
                                + np.einsum('i, ij, ijk, j->jk', c_ws, u_val1, p_gphi_c[..., 1], cell_measure))  # (NC,cldof)
            cell_int1 = -(np.einsum('i, ij, ijk, j->jk', c_ws, nolinear_val0, p_gphi_c[..., 0], cell_measure)
                          + np.einsum('i, ij, ijk, j->jk', c_ws, nolinear_val1, p_gphi_c[..., 1], cell_measure))  # (NC,cldof)
            cell_int2 = (np.einsum('i, ij, ijk, j->jk', c_ws, f_val[..., 0], p_gphi_c[..., 0], cell_measure)
                         + np.einsum('i, ij, ijk, j->jk', c_ws, f_val[..., 1], p_gphi_c[..., 1], cell_measure))  # (NC,cldof)

            np.add.at(prv, Dir_face2dof, dir_int0)
            np.add.at(prv, Dir_cell2dof, dir_int1)
            np.add.at(prv, pcell2dof, cell_int0 + cell_int1 + cell_int2)

            # # Method I: The following code is right! Pressure satisfies \int_\Omega p = 0
            plsm_temp = bmat([[plsm, basis_int.reshape(-1, 1)], [basis_int, None]], format='csr')
            prv = np.r_[prv, 0]
            ph[:] = spsolve(plsm_temp, prv)[:-1]  # we have added one addtional dof

            # # Method II: Using the Dirichlet boundary of pressure
            # def dir_pressure(p):
            #     return pde.pressure(p, next_t)
            # bc = DirichletBC(pspace, dir_pressure)
            # plsm_temp, prv = bc.apply(plsm.copy(), prv)
            # ph[:] = spsolve(plsm_temp, prv).reshape(-1)

            # # --- to update the velocity value --- # #
            grad_ph = pspace.grad_value(ph, c_bcs)  # (NQ,NC,2)

            # the velocity u's Left-Matrix
            ulm0 = 1 / dt * ulmm + pde.nu * ulsm
            ulm1 = 1 / dt * ulmm + pde.nu * ulsm

            # # to get the u's Right-hand Vector
            def dir_u0(p):
                return pde.dirichlet(p, next_t)[..., 0]

            def dir_u1(p):
                return pde.dirichlet(p, next_t)[..., 1]

            # for the first-component of velocity
            urv0 = np.zeros((vdof.number_of_global_dofs(),), dtype=self.ftype)  # (Nvdof,)
            urv0_c = np.einsum('i, ij, ijk, j->jk', c_ws, u_val0/dt - grad_ph[..., 0] - nolinear_val0
                               + f_val[..., 0], u_phi, cell_measure)  # (NC,clodf)
            urv0_f = self.pde.nu * np.einsum('i, ij, ijn, j->jn', f_ws, Neumann_0, u_phi_f, neu_face_measure)
            np.add.at(urv0, ucell2dof, urv0_c)
            # np.add.at(urv0, Neu_face2dof, urv0_f)
            u0_bc = DirichletBC(vspace, dir_u0, threshold=idxDirEdge)
            ulm0, urv0 = u0_bc.apply(ulm0, urv0)
            uh0[:] = spsolve(ulm0, urv0).reshape(-1)

            # for the second-component of velocity
            urv1 = np.zeros((vdof.number_of_global_dofs(),), dtype=self.ftype)  # (Nvdof,)
            urv1_c = np.einsum('i, ij, ijk, j->jk', c_ws, u_val1/dt - grad_ph[..., 1] - nolinear_val1
                               + f_val[..., 1], u_phi, cell_measure)  # (NC,clodf)
            urv1_f = self.pde.nu * np.einsum('i, ij, ijn, j->jn', f_ws, Neumann_1, u_phi_f, neu_face_measure)
            np.add.at(urv1, ucell2dof, urv1_c)
            # np.add.at(urv1, Neu_face2dof, urv1_f)
            u1_bc = DirichletBC(vspace, dir_u1, threshold=idxDirEdge)
            ulm1, urv1 = u1_bc.apply(ulm1, urv1)
            uh1[:] = spsolve(ulm1, urv1).reshape(-1)

            if nt % 50 == 0:
                print('# ------------ logging the circle info ------------ #')
                print('current t = ', currt_t)
                p_l2err, u0_l2err, u1_l2err = self.currt_error(ph, uh0, uh1, next_t)
                print('p_l2err = %e,  u0_l2err = %e,  u1_l2err = %e' % (p_l2err, u0_l2err, u1_l2err))
                print('# ------------------------------------------------- # \n')
                if np.isnan(p_l2err) | np.isnan(u0_l2err) | np.isnan(u1_l2err):
                    print('Some error is nan: breaking the program')
                    break

            # print('end of current time')

        print('# ------------ the end error ------------ #')
        u_l2err, u_h1err, p_l2err = self.currt_error(ph, uh0, uh1, timemesh[-1])
        print('u_l2err = %e,  u_h1err = %e,  p_l2err = %e' % (u_l2err, u_h1err, p_l2err))
        return u_l2err, u_h1err, p_l2err

    @timer
    def NS_VC_Solver_T2ndOrder(self):
        """
        The Navier-Stokes Velocity-Correction scheme solver.
        """
        pde = self.pde
        dt = self.dt
        timemesh = self.timemesh
        uh0 = self.uh0
        uh1 = self.uh1
        ph = self.ph
        vspace = self.vspace
        pspace = self.pspace
        vdof = self.vdof
        pdof = self.pdof
        pface2dof = pdof.face_to_dof()  # (NE,fldof)
        pcell2dof = pdof.cell_to_dof()  # (NC,cldof)
        ucell2dof = vspace.cell_to_dof()  # (NC,cldof)
        vface2dof = vdof.face_to_dof()  # (NE,fldof)

        idxDirEdge = self.set_Dirichlet_edge()
        cellidxDir = self.mesh.ds.edge2cell[idxDirEdge, 0]
        localidxDir = self.mesh.ds.edge2cell[idxDirEdge, 2]

        Dir_face2dof = pface2dof[idxDirEdge, :]  # (NDir,flodf)
        Dir_cell2dof = pcell2dof[cellidxDir, :]  # (NDir,cldof)
        n_Dir = self.mesh.face_unit_normal(index=idxDirEdge)  # (NDir,2)
        dir_face_measure = self.mesh.entity_measure('face', index=idxDirEdge)  # (NDir,2)
        cell_measure = self.mesh.cell_area()
        dir_cell_measure = self.mesh.cell_area(cellidxDir)

        idxNeuEdge = self.mesh.ds.boundary_face_index()  # here Neumann edges are all the boundary
        n_Neu = self.mesh.face_unit_normal(index=idxNeuEdge)  # (NNeu,2)
        neu_face_measure = self.mesh.entity_measure('face', index=idxNeuEdge)  # (NNeu,2)
        Neu_face2dof = vface2dof[idxNeuEdge, :]  # (NNeu,flodf)

        f_q = self.integralalg.faceintegrator
        f_bcs, f_ws = f_q.get_quadrature_points_and_weights()  # f_bcs.shape: (NQ,(GD-1)+1)
        f_pp = self.mesh.bc_to_point(f_bcs, index=idxDirEdge)  # f_pp.shape: (NQ,NDir,GD) the physical Gauss points
        f_pp_Neu = self.mesh.bc_to_point(f_bcs, index=idxNeuEdge)  # f_pp.shape: (NQ,NNeu,GD) the physical Gauss points

        c_q = self.integralalg.cellintegrator
        c_bcs, c_ws = c_q.get_quadrature_points_and_weights()  # c_bcs.shape: (NQ,GD+1)
        c_pp = self.mesh.bc_to_point(c_bcs)  # c_pp.shape: (NQ_cell,NC,GD) the physical Gauss points

        # # t^{n+1}: Pressure-Left-StiffMatrix
        plsm = self.pspace.stiff_matrix()
        basis_int = pspace.integral_basis()
        p_phi = pspace.face_basis(f_bcs)  # (NQ,1,fldof). 实际上这里可以直接用 pspace.basis(f_bcs), 两个函数的代码是相同的
        p_gphi_f = pspace.edge_grad_basis(f_bcs, cellidxDir, localidxDir)  # (NDir,NQ,cldof,GD)
        p_gphi_c = pspace.grad_basis(c_bcs)  # (NQ_cell,NC,ldof,GD)

        # # t^{n+1}: Velocity-Left-MassMatrix and -StiffMatrix
        ulmm = self.vspace.mass_matrix()
        ulsm = self.vspace.stiff_matrix()
        u_phi = vspace.basis(c_bcs)  # (NQ,1,cldof)
        u_phi_f = vspace.face_basis(f_bcs)  # (NQ,1,fldof) 实际上这里可以直接用 vspace.basis(f_bcs), 两个函数的代码是相同的

        # # some settings
        last_uh0 = vspace.function()
        last_uh1 = vspace.function()
        uh_hat_0 = vspace.function()
        uh_hat_1 = vspace.function()
        uh_star_0 = vspace.function()
        uh_star_1 = vspace.function()

        def init_vel(p):
            return self.pde.velocity(p, 0)
        vel_temp = vspace.interpolation(init_vel)
        last_uh0[:] = vel_temp[..., 0]
        last_uh1[:] = vel_temp[..., 1]

        NT = len(timemesh)
        for nt in range(NT - 1):
            currt_t = timemesh[nt]
            next_t = currt_t + dt

            # ---------------------------------------
            # 1st-step: get the p^{n+1}
            # ---------------------------------------
            # # Pressure-Right-Matrix
            if nt == 0:
                coeff = 1./dt
                uh_hat_0[:] = last_uh0[:]
                uh_hat_1[:] = last_uh1[:]
                uh_star_0[:] = last_uh0[:]
                uh_star_1[:] = last_uh1[:]
            else:
                coeff = 3./(2*dt)
                uh_hat_0[:] = 2*uh0 - 1/2 * last_uh0
                uh_hat_1[:] = 2*uh1 - 1/2 * last_uh1
                uh_star_0[:] = 2*uh0 - last_uh0
                uh_star_1[:] = 2*uh1 - last_uh1

                last_uh0[:] = uh0[:]
                last_uh1[:] = uh1[:]

            # for Dirichlet-face-integration
            gu_val0 = self.uh_grad_value_at_faces(uh_star_0, f_bcs, cellidxDir, localidxDir)  # grad_u0: (NQ,NDir,GD)
            gu_val1 = self.uh_grad_value_at_faces(uh_star_1, f_bcs, cellidxDir, localidxDir)  # grad_u0: (NQ,NDir,GD)
            curl_uh_dir = gu_val1[..., 0] - gu_val0[..., 1]  # (NQ,NDir)

            # for cell-integration
            u_val0 = vspace.value(uh_hat_0, c_bcs)  # (NQ,NC)
            u_val1 = vspace.value(uh_hat_1, c_bcs)

            nolinear_val = self.NSNolinearTerm(uh_star_0, uh_star_1, c_bcs)  # last_nolinear_val.shape: (NQ,NC,GD)
            nolinear_val0 = nolinear_val[..., 0]  # (NQ,NC)
            nolinear_val1 = nolinear_val[..., 1]  # (NQ,NC)

            # # other settings
            Neumann_0 = pde.neumann_0(f_pp_Neu, next_t, n_Neu)  # (NQ,NE)
            Neumann_1 = pde.neumann_1(f_pp_Neu, next_t, n_Neu)  # (NQ,NE)
            uDir_val = pde.dirichlet(f_pp, next_t)  # (NQ,NDir,GD)
            f_val = pde.source(c_pp, next_t)  # (NQ,NC,GD)

            # # --- to update the pressure value --- # #
            # # to get the Pressure's Right-hand Vector
            prv = np.zeros((pdof.number_of_global_dofs(),), dtype=self.ftype)  # (Npdof,)

            # for Dirichlet faces integration
            dir_int0 = -1 * coeff * np.einsum('i, ijk, jk, ijn, j->jn', f_ws, uDir_val, n_Dir, p_phi,
                                              dir_face_measure)  # (NDir,fldof)
            dir_int1 = - pde.nu * (np.einsum('i, j, ij, jin, j->jn', f_ws, n_Dir[:, 1], curl_uh_dir,
                                             p_gphi_f[..., 0], dir_face_measure)
                                   + np.einsum('i, j, ij, jin, j->jn', f_ws, -n_Dir[:, 0], curl_uh_dir,
                                               p_gphi_f[..., 1], dir_face_measure))  # (NDir,cldof)
            # for cell integration
            cell_int = np.einsum('i, ij, ijk, j->jk', c_ws, 1/dt*u_val0 - nolinear_val0 + f_val[..., 0], p_gphi_c[..., 0], cell_measure) \
                       + np.einsum('i, ij, ijk, j->jk', c_ws, 1/dt*u_val1 - nolinear_val1 + f_val[..., 1], p_gphi_c[..., 1], cell_measure)  # (NC,cldof)

            np.add.at(prv, Dir_face2dof, dir_int0)
            np.add.at(prv, Dir_cell2dof, dir_int1)
            np.add.at(prv, pcell2dof, cell_int)

            # # Method I: The following code is right! Pressure satisfies \int_\Omega p = 0
            plsm_temp = bmat([[plsm, basis_int.reshape(-1, 1)], [basis_int, None]], format='csr')
            prv = np.r_[prv, 0]
            ph[:] = spsolve(plsm_temp, prv)[:-1]  # we have added one addtional dof

            # # Method II: Using the Dirichlet boundary of pressure
            # def dir_pressure(p):
            #     return pde.pressure(p, next_t)
            # bc = DirichletBC(pspace, dir_pressure)
            # plsm_temp, prv = bc.apply(plsm.copy(), prv)
            # ph[:] = spsolve(plsm_temp, prv).reshape(-1)

            # # --- to update the velocity value --- # #
            grad_ph = pspace.grad_value(ph, c_bcs)  # (NQ,NC,2)

            # the velocity u's Left-Matrix
            ulm0 = coeff * ulmm + pde.nu * ulsm
            ulm1 = coeff * ulmm + pde.nu * ulsm

            # # to get the u's Right-hand Vector
            def dir_u0(p):
                return pde.dirichlet(p, next_t)[..., 0]

            def dir_u1(p):
                return pde.dirichlet(p, next_t)[..., 1]

            # for the first-component of velocity
            urv0 = np.zeros((vdof.number_of_global_dofs(),), dtype=self.ftype)  # (Nvdof,)
            urv0_c = np.einsum('i, ij, ijk, j->jk', c_ws, u_val0 / dt - grad_ph[..., 0] - nolinear_val0
                               + f_val[..., 0], u_phi, cell_measure)  # (NC,clodf)
            np.add.at(urv0, ucell2dof, urv0_c)
            # # urv0_f = self.pde.nu * np.einsum('i, ij, ijn, j->jn', f_ws, Neumann_0, u_phi_f, neu_face_measure)
            # # np.add.at(urv0, Neu_face2dof, urv0_f)
            u0_bc = DirichletBC(vspace, dir_u0, threshold=idxDirEdge)
            ulm0, urv0 = u0_bc.apply(ulm0, urv0)
            uh0[:] = spsolve(ulm0, urv0).reshape(-1)

            # for the second-component of velocity
            urv1 = np.zeros((vdof.number_of_global_dofs(),), dtype=self.ftype)  # (Nvdof,)
            urv1_c = np.einsum('i, ij, ijk, j->jk', c_ws, u_val1 / dt - grad_ph[..., 1] - nolinear_val1
                               + f_val[..., 1], u_phi, cell_measure)  # (NC,clodf)
            urv1_f = self.pde.nu * np.einsum('i, ij, ijn, j->jn', f_ws, Neumann_1, u_phi_f, neu_face_measure)
            np.add.at(urv1, ucell2dof, urv1_c)
            # np.add.at(urv1, Neu_face2dof, urv1_f)
            u1_bc = DirichletBC(vspace, dir_u1, threshold=idxDirEdge)
            ulm1, urv1 = u1_bc.apply(ulm1, urv1)
            uh1[:] = spsolve(ulm1, urv1).reshape(-1)

            if nt % max([int(NT / 5), 1]) == 0:
                print('# ------------ logging the circle info ------------ #')
                print('current t = ', currt_t)
                p_l2err, u0_l2err, u1_l2err = self.currt_error(ph, uh0, uh1, next_t)
                print('p_l2err = %e,  u0_l2err = %e,  u1_l2err = %e' % (p_l2err, u0_l2err, u1_l2err))
                print('# ------------------------------------------------- # \n')
                if np.isnan(p_l2err) | np.isnan(u0_l2err) | np.isnan(u1_l2err):
                    print('Some error is nan: breaking the program')
                    break

            # print('end of current time')

        print('# ------------ the end error ------------ #')
        u_l2err, u_h1err, p_l2err = self.currt_error(ph, uh0, uh1, timemesh[-1])
        print('u_l2err = %e,  u_h1err = %e,  p_l2err = %e' % (u_l2err, u_h1err, p_l2err))
        return u_l2err, u_h1err, p_l2err

    def currt_error(self, ph, uh0, uh1, t):
        pde = self.pde

        def currt_pressure(p):
            return pde.pressure(p, t)
        p_l2err = self.pspace.integralalg.L2_error(currt_pressure, ph)

        def currt_u0(p):
            return pde.velocity(p, t)[..., 0]
        u0_l2err = self.vspace.integralalg.L2_error(currt_u0, uh0)

        def currt_u1(p):
            return pde.velocity(p, t)[..., 1]
        u1_l2err = self.vspace.integralalg.L2_error(currt_u1, uh1)

        def currt_grad_velocity0(p):
            return pde.grad_velocity0(p, t)
        u0_h1err = self.vspace.integralalg.L2_error(currt_grad_velocity0, uh0.grad_value)

        def currt_grad_velocity1(p):
            return pde.grad_velocity1(p, t)
        u1_h1err = self.vspace.integralalg.L2_error(currt_grad_velocity1, uh1.grad_value)

        return np.sqrt(u0_l2err**2 + u1_l2err**2), np.sqrt(u0_h1err**2 + u1_h1err**2), p_l2err

    def uh_grad_value_at_faces(self, vh, f_bcs, cellidx, localidx):
        cell2dof = self.vdof.cell2dof
        f_gphi = self.vspace.edge_grad_basis(f_bcs, cellidx, localidx)  # (NE,NQ,cldof,GD)

        # val.shape: (NQ,NE,GD)
        # vh.shape: (v_gdof,)
        # vh[cell2dof[cellidx]].shape: (Ncellidx,cldof)
        val = np.einsum('ik, ijkm->jim', vh[cell2dof[cellidx]], f_gphi)
        return val

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























