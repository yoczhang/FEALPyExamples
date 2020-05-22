#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: HHOStokesSpace2d.py
# @Author: Yongchao Zhang
# @E-mail: yoczhang@126.com
# @Time: May 19, 2020
# ---


import numpy as np
from numpy.linalg import inv
from fealpy.common import block, block_diag
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye, bmat

from fealpy.functionspace.function import Function
from fealpy.quadrature import GaussLegendreQuadrature
from fealpy.quadrature import PolygonMeshIntegralAlg
# from fealpy.functionspace.ScaledMonomialSpace2d import SMDof2d, ScaledMonomialSpace2d
from myScaledMonomialSpace2d import SMDof2d, ScaledMonomialSpace2d
from HHOScalarSpace2d import HHODof2d, HHOScalarSpace2d


class HHOStokesDof2d:
    def __init__(self, mesh, p):
        self.mesh = mesh
        self.p = p
        self.vDof = self.velocityDof()
        self.pDof = self.pressureDof()

    def velocityDof(self):
        # # note that, this Dof only has the scalar Dof
        return HHODof2d(self.mesh, self.p)

    def pressureDof(self):
        return SMDof2d(self.mesh, self.p)

    def number_of_global_dofs(self):
        return 2*self.vDof.number_of_global_dofs() + self.pDof.number_of_global_dofs()


class HHOStokesSapce2d:
    def __init__(self, mesh, p):
        self.mesh = mesh
        self.p = p
        self.dof = HHOStokesDof2d(mesh, p)
        self.vSpace = HHOScalarSpace2d(mesh, p)
        self.pSpace = ScaledMonomialSpace2d(mesh, p)
        self.integralalg = self.vSpace.integralalg

    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def system_matrix(self, nu):
        A = self.velocity_matrix()  # (2*vgdof,2*vgdof)
        B = self.divergence_matrix()  # (pgdof,2*vgdof)
        P = self.pressure_correction()  # (1,2*vgdof+pgdof)

        S0 = bmat([[nu*A, B.T], [B, None]], format='csr')
        S = bmat([[S0, P.T], [P, None]], format='csr')  # (2*vgdof+pgdof+1, 2*vgdof+pgdof+1)
        return S

    def system_source(self, f):
        pgdof = self.pSpace.number_of_global_dofs()
        vs = self.velocity_source(f)
        z0 = np.zeros((pgdof+1, 1), dtype=np.float)
        # return bmat([[vs], [z0]], format='csr')
        return np.concatenate([vs, z0])

    def velocity_matrix(self):
        scalarM = self.vSpace.system_matrix()  # (vgdof,vgdof), here, vgdof is the number of dofs for Scalar hho-variable

        velocityM = bmat([[scalarM, None], [None, scalarM]], format='csr')  # (2*vgdof,2*vgdof)
        return velocityM

    def velocity_source(self, f):
        vgdof = self.vSpace.dof.number_of_global_dofs()
        fh = self.source_vector(f)  # (NC,ldof,2)
        fh1 = fh[..., 0]  # (NC,ldof)
        fh2 = fh[..., 1]  # (NC,ldof)
        shape = fh1.shape
        v1 = np.zeros([vgdof, 1], dtype=np.float)
        v2 = np.zeros([vgdof, 1], dtype=np.float)
        v1[:(shape[0]*shape[1]), 0] = fh1.flatten()
        v2[:(shape[0] * shape[1]), 0] = fh2.flatten()

        sourceV = np.concatenate([v1, v2])  # (2*vgdof,1)
        return sourceV

    def source_vector(self, f):
        phi = self.vSpace.basis  # basis is inherited from class ScaledMonomialSpace2d()

        def u(x, index):
            return np.einsum('ijn, ijm->ijmn', f(x), phi(x, index=index))
            # # f(x).shape: (NQ,NC,2).    phi(x,...).shape: (NQ,NC,ldof)

        fh = self.integralalg.integral(u, celltype=True)  # (NC,ldof,2)
        return fh

    def divergence_matrix(self):
        NC = self.mesh.number_of_cells()
        vldof = self.vSpace.smldof
        pldof = self.pSpace.number_of_local_dofs()
        eldof = self.p + 1
        vgdof = self.vSpace.dof.number_of_global_dofs()  # number of all dofs, contains edge-dofs and cell-dofs
        pgdof = NC*pldof
        cell2dof, doflocation = self.vSpace.dof.cell_to_dof()
        cell2dof_split = np.hsplit(cell2dof, doflocation[1:-1])

        divM0, divM1 = self.cell_divergence_matrix()
        divM0_split = np.hsplit(divM0, doflocation[1:-1])
        divM1_split = np.hsplit(divM1, doflocation[1:-1])

        def f(x):
            divM0_C = x[0]
            divM1_C = x[1]
            dof_C = x[2]  # (NCdof,)
            Cidx = x[3]  # the index of the current cell
            Ndof_C = len(dof_C)

            # --- get the row and col index --- #
            ro = range(Cidx*pldof, (Cidx+1)*pldof)
            rowIndex = np.einsum('i, k->ik', ro, np.ones(2*Ndof_C,))
            colIndex = np.einsum('i, k->ik', np.ones(len(ro),), dof_C)
            colIndex = np.concatenate([colIndex, vgdof+colIndex], axis=1)

            # --- add to the global matrix and vector --- #
            divM_C = np.concatenate([divM0_C, divM1_C], axis=1)
            r = csr_matrix((divM_C.flat, (rowIndex.flat, colIndex.flat)), shape=(pgdof, 2*vgdof), dtype=np.float)
            return r
        divM = sum(list(map(f, zip(divM0_split, divM1_split, cell2dof_split, range(NC)))))
        return divM  # (pgdof,2*vgdof)

    def cell_divergence_matrix(self):
        p = self.p
        vSpace = self.vSpace
        mesh = self.mesh
        vldof = self.vSpace.smldof
        pldof = self.pSpace.number_of_local_dofs()
        eldof = p + 1
        NC = mesh.number_of_cells()
        node = mesh.entity('node')
        edge = mesh.entity('edge')
        edge2cell = mesh.ds.edge_to_cell()
        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
        n = mesh.edge_normal()  # (NE,2), the normal vector of edges (the length of this normal is the edge-length)
        # # The direction of normal vector is from edge2cell[i,0] to edge2cell[i,1]
        # # (that is, from the cell with smaller number to the cell with larger number).

        qf = GaussLegendreQuadrature(p + 3)
        bcs, ws = qf.quadpts, qf.weights  # bcs.shape: (NQ,2); ws.shape: (NQ,)
        ps = np.einsum('ij, kjm->ikm', bcs, node[edge])  # (NQ,NE,2), NE is the number of edges

        # --- the basis values at ps --- #
        # # phi0, phi1 are the potential variable, are trial functions, taking order p,
        # # pphi0, pphi1 are the test functions, taking order p+1.
        # # So, in the following,
        # # smldof denotes the number of local dofs in smspace in order p,
        # # psmldof denotes the number of local dofs in smspace in order p+1.
        # #
        vphi0 = self.vSpace.basis(ps, index=edge2cell[:, 0])  # (NQ,NE,smldof)
        vphi1 = self.vSpace.basis(ps[:, isInEdge, :], index=edge2cell[isInEdge, 1])
        pphi0 = self.pSpace.basis(ps, index=edge2cell[:, 0])  # (NQ,NE,smldof)
        pphi1 = self.pSpace.basis(ps[:, isInEdge, :], index=edge2cell[isInEdge, 1])
        vephi = self.vSpace.edge_basis(ps)  # (NQ,NE,eldof), eldof is the number of local 1D dofs on one edge

        # --- construct the matrix --- #
        vcell2dof = vSpace.dof.cell2dof  # cell2dof.shape: (\sum_C{Cldof},)
        vcell2dofLocation = vSpace.dof.cell2dofLocation
        divM0 = np.zeros((pldof, vcell2dofLocation[-1]), dtype=np.float)  # (pldof,\sum_C{Cldof}), Cldof is the number of dofs in one cell
        divM1 = np.zeros((pldof, vcell2dofLocation[-1]), dtype=np.float)  # (pldof,\sum_C{Cldof}), Cldof is the number of dofs in one cell
        T0 = np.zeros((NC, pldof, vldof), dtype=np.float)  # (NC,pldof,vldof)
        T1 = np.zeros((NC, pldof, vldof), dtype=np.float)  # (NC,pldof,vldof)

        # --- the body divergence matrix, ((u0,u1), \nabla q)_T = (u0, \partial_x q)_T + (u1, \partial_y q)_T
        def f(x, index):
            pgphi = self.pSpace.grad_basis(x, index=index)  # using the cell-integratipon, so pgphi: (NQ,NC,pldof,2)
            vphi = self.vSpace.basis(x, index=index)  # using the cell-integration, so vphi: (NQ,NC,vldof)
            return np.einsum('...k, ...mn->...mkn', vphi, pgphi)
        divcell = self.integralalg.integral(f, celltype=True)  # (NC,pldof,vldof,2)
        # def f(x, index):
        #     vgphi = self.vSpace.grad_basis(x, index=index)  # using the cell-integration, so vgphi: (NQ,NC,vldof,2)
        #     pphi = self.pSpace.basis(x, index=index)  # using the cell-integratipon, so pphi: (NQ,NC,pldof)
        #     return np.einsum('...mn, ...k->...kmn', vgphi, pphi)
        # divcell = self.integralalg.integral(f, celltype=True)  # (NC,pldof,vldof,2)

        np.add.at(T0, np.arange(NC), -divcell[..., 0])  # T0.shape: (NC,pldof,vldof)
        np.add.at(T1, np.arange(NC), -divcell[..., 1])  # T1.shape: (NC,pldof,vldof)

        # # --- edge integration. Part I: (-u_T*n_0, q)_{\partial T} and (-u_T*n_1, q)_{\partial T}
        # T_0 = np.einsum('i, ijk, ijm, jn->jmkn', ws, vphi0, pphi0, n)  # (NE,pldof,vldof,2)
        # T_1 = np.einsum('i, ijk, ijm, jn->jmkn', ws, vphi1, pphi1, -n[isInEdge, :])  # (NInE,pldof,vldof,2)
        # np.add.at(T0, edge2cell[:, 0], -T_0[..., 0])
        # np.add.at(T0, edge2cell[isInEdge, 1], -T_1[..., 0])
        # np.add.at(T1, edge2cell[:, 0], -T_0[..., 1])
        # np.add.at(T1, edge2cell[isInEdge, 1], -T_1[..., 1])

        idx = vcell2dofLocation[0:-1].reshape(-1, 1) + np.arange(vldof)  # (NC,vldof)
        divM0[:, idx] = T0.swapaxes(0, 1)  # divM0.shape: (pldof,\sum_C{Cldof})
        divM1[:, idx] = T1.swapaxes(0, 1)  # divM1.shape: (pldof,\sum_C{Cldof})

        # --- edge integration. Part II: (v_F, \nabla w\cdot n)_{\partial T}
        F_0 = np.einsum('i, ijk, ijm, jn->mjkn', ws, vephi, pphi0, n)  # (pldof,NE,eldof,2)
        F_1 = np.einsum('i, ijk, ijm, jn->mjkn', ws, vephi[:, isInEdge, :], pphi1, -n[isInEdge, :])  # (pldof,NInE,eldof,2)
        idx = vcell2dofLocation[edge2cell[:, [0]]] + edge2cell[:, [2]] * eldof + np.arange(eldof)  # (NE,eldof)
        idx += vldof  # rearrange the dofs
        divM0[:, idx] = F_0[..., 0]  # (pldof,\sum_C{Cldof}), Cldof is the number of dofs in one cell
        divM1[:, idx] = F_0[..., 1]  # (pldof,\sum_C{Cldof}), Cldof is the number of dofs in one cell
        idx = vcell2dofLocation[edge2cell[isInEdge, 1]].reshape(-1, 1) + \
              eldof * edge2cell[isInEdge, 3].reshape(-1, 1) + np.arange(eldof)  # (NInE,eldof)
        idx += vldof  # rearrange the dofs
        divM0[:, idx] = F_1[..., 0]  # (pldof,\sum_C{Cldof}), Cldof is the number of dofs in one cell
        divM1[:, idx] = F_1[..., 1]  # (pldof,\sum_C{Cldof}), Cldof is the number of dofs in one cell

        return -divM0, -divM1

    def pressure_correction(self):
        p = self.p
        pgdof = self.pSpace.number_of_global_dofs()
        vgdof = self.vSpace.number_of_global_dofs()

        pphi = self.pSpace.basis  # (NQ,NC,pldof)
        pIn = self.integralalg.integral(pphi, celltype=True)  # (NC,pldof)
        pIn = pIn.reshape(-1, 1)  # (pgdof,1)

        r = np.zeros((1, 2*vgdof), dtype=np.float)
        r = np.concatenate([r, pIn.T], axis=1)  # (1,2*vgdof+pgdof)
        return r

    def velocity_project(self, velocity):  # here, the velocity must be (u1, u2)
        uh = self.vSpace.project(velocity, dim=2)
        # # (vgdof,2), uh[:,0] is the project of u1, and uh[:,1] is the project of u2
        return uh

    def pressure_project(self, pressure):
        pspace = self.pSpace
        invCM = inv(pspace.cell_mass_matrix())  # (NC,smldof,smldof), smldof is the number of local dofs of smspace

        ph = pspace.function()  # (pgdof,)
        phi = pspace.basis

        def f1(x, index):
            return np.einsum('..., ...m->...m', pressure(x), phi(x, index))
        b = self.integralalg.integral(f1, celltype=True)

        ph[:] = (invCM @ b[:, :, np.newaxis]).flatten()
        return ph



    # def function(self, dim=None, array=None):
    #     f = Function(self, dim=dim, array=array)
    #     return f
    #
    # def array(self, dim=None):
    #     vgdof = self.vSpace.number_of_global_dofs()
    #     # # So, here, only used for the velocity variable
    #     gdof = 2*vgdof
    #     if dim in {None, 1}:
    #         shape = gdof
    #     elif type(dim) is int:
    #         shape = (gdof, dim)
    #     elif type(dim) is tuple:
    #         shape = (gdof, ) + dim
    #     return np.zeros(shape, dtype=self.ftype)







