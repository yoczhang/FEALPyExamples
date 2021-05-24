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
# from fealpy.common import block, block_diag
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye, bmat
# from fealpy.decorator import cartesian

# from fealpy.functionspace.function import Function
from fealpy.quadrature import GaussLegendreQuadrature
# from fealpy.quadrature import PolygonMeshIntegralAlg
from fealpy.functionspace.ScaledMonomialSpace2d import SMDof2d, ScaledMonomialSpace2d
# from myScaledMonomialSpace2d import SMDof2d, ScaledMonomialSpace2d
from HHOScalarSpace2d import HHODof2d, HHOScalarSpace2d
from scipy.special import comb, perm


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

    def number_of_velocity_dofs(self):
        return 2 * self.vDof.number_of_global_dofs()

    def number_of_pressure_dofs(self):
        return self.pDof.number_of_global_dofs()


class HHOStokesSpace2d:
    def __init__(self, mesh, p):
        self.mesh = mesh
        self.p = p
        self.dof = HHOStokesDof2d(mesh, p)
        self.vSpace = HHOScalarSpace2d(mesh, p)
        self.pSpace = ScaledMonomialSpace2d(mesh, p)
        self.integralalg = self.vSpace.integralalg

    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def number_of_velocity_dofs(self):
        return self.dof.number_of_velocity_dofs()

    def number_of_pressure_dofs(self):
        return self.dof.number_of_pressure_dofs()

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

    def velocity_stabilizer_matrix(self):
        scalar_stabM = self.vSpace.stabilizer_matrix()  # (vgdof,vgdof), here, vgdof is the number of dofs for Scalar hho-variable

        velocity_stabM = bmat([[scalar_stabM, None], [None, scalar_stabM]], format='csr')  # (2*vgdof,2*vgdof)
        return velocity_stabM

    def velocity_source(self, f):
        vgdof = self.vSpace.dof.number_of_global_dofs()
        phi = self.vSpace.basis  # basis is inherited from class ScaledMonomialSpace2d()

        def u(x, index):
            # # f(x).shape: (NQ,NC,2).    phi(x,...).shape: (NQ,NC,ldof)
            return np.einsum('ijn, ijm->ijmn', f(x), phi(x, index=index))
        fh = self.integralalg.integral(u, celltype=True)  # (NC,ldof,2)
        fh1 = fh[..., 0]  # (NC,ldof)
        fh2 = fh[..., 1]  # (NC,ldof)
        shape = fh1.shape
        v1 = np.zeros([vgdof, 1], dtype=np.float)
        v2 = np.zeros([vgdof, 1], dtype=np.float)
        v1[:(shape[0]*shape[1]), 0] = fh1.flatten()
        v2[:(shape[0] * shape[1]), 0] = fh2.flatten()

        sourceV = np.concatenate([v1, v2])  # (2*vgdof,1)
        return sourceV

    def divergence_matrix(self):
        NC = self.mesh.number_of_cells()
        pldof = self.pSpace.number_of_local_dofs()
        vgdof = self.vSpace.dof.number_of_global_dofs()  # number of all dofs, contains edge-dofs and cell-dofs
        pgdof = NC*pldof
        cell2dof, doflocation = self.vSpace.dof.cell_to_dof()
        cell2dof_split = np.hsplit(cell2dof, doflocation[1:-1])

        divM0, divM1 = self.cell_divergence_matrix()  # divM0: (pldof,\sum_C{Cldof}), Cldof is the number of dofs in one cell
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

    def cell_divergence_matrix(self):  # reference: (book) The Hybrid High-Order Method for Polytopal Meshes.
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

        np.add.at(T0, np.arange(NC), -divcell[..., 0])  # T0.shape: (NC,pldof,vldof)
        np.add.at(T1, np.arange(NC), -divcell[..., 1])  # T1.shape: (NC,pldof,vldof)

        idx = vcell2dofLocation[0:-1].reshape(-1, 1) + np.arange(vldof)  # (NC,vldof)
        divM0[:, idx] = T0.swapaxes(0, 1)  # divM0.shape: (pldof,\sum_C{Cldof})
        divM1[:, idx] = T1.swapaxes(0, 1)  # divM1.shape: (pldof,\sum_C{Cldof})

        # --- edge integration: (v_F, \nabla w\cdot n)_{\partial T}
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
        vgdof = self.vSpace.number_of_global_dofs()

        pphi = self.pSpace.basis  # (NQ,NC,pldof)
        intp = self.integralalg.integral(pphi, celltype=True)  # (NC,pldof)
        intp = intp.reshape(-1, 1)  # (pgdof,1)

        r = np.zeros((1, 2*vgdof), dtype=np.float)
        r = np.concatenate([r, intp.T], axis=1)  # (1,2*vgdof+pgdof)
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

    def velocity_L2_error(self, velocity, uh0, uh1, celltype=False):
        """
        :param velocity: is true solution velocity and from pde.velocity
        :param uh0: is the numerical solution uh0
        :param uh1: is the numerical solution uh1
        :param celltype:
        :return:
        """
        uI = self.velocity_project(velocity)
        uI0 = uI[:, 0]
        uI1 = uI[:, 1]
        err0 = self.vSpace.L2_error(uI0, uh0, celltype=celltype)
        err1 = self.vSpace.L2_error(uI1, uh1, celltype=celltype)
        return np.sqrt(err0**2 + err1**2)

    def pressure_L2_error(self, pressure, ph, celltype=False):
        """
        :param pressure: is true solution pressure and from pde.pressure
        :param ph: is the numerical solution ph
        :param celltype:
        :return:
        """
        pI = self.pressure_project(pressure)
        ep = pI - ph

        def f(x, index=np.s_[:]):
            evalue = self.pSpace.value(ep, x, index=index)  # the evalue has the same shape of x.
            return evalue*evalue
        err = self.integralalg.integral(f, celltype=celltype)
        return np.sqrt(err)

    def velocity_energy_error(self, velocity, A, uh0, uh1):
        """
        :param velocity: is true solution velocity and from pde.velocity
        :param A: is the system matrix which has been treated with the Dirichlet boundary condition !!!
        :param uh0: is the numerical solution uh0
        :param uh1: is the numerical solution uh1
        :param celltype:
        :return:
        """

        vgdof = self.vSpace.number_of_global_dofs()
        uI = self.velocity_project(velocity)
        uI0 = uI[:, 0]
        uI1 = uI[:, 1]

        eu0 = uI0 - uh0
        eu1 = uI1 - uh1
        eu = np.concatenate([eu0, eu1])
        eA = A[:2*vgdof, :2*vgdof]
        energyerror = np.sqrt(eu@eA@eu)

        return energyerror

    def applyDirichletBC(self, A, b, dirichletfunc, idxDirEdge=None, StaticCondensation=False):
        p = self.p
        mesh = self.mesh
        uD = dirichletfunc  # uD(bcs): (NQ,NC,ldof,2)
        vgdof = self.vSpace.number_of_global_dofs()
        pgdof = self.pSpace.number_of_global_dofs()
        idxDirEdge = self.defaultDirichletEdges() if idxDirEdge is None else idxDirEdge
        idxDirDof = self.setStokesDirichletDofs(idxDirEdge=idxDirEdge, StaticCondensation=StaticCondensation)
        node = mesh.entity('node')
        edge = mesh.entity('edge')
        hE = mesh.edge_length()

        # --- --- #
        # Nalldof = 2*vgdof+pgdof+1
        # assert Nalldof == len(b), 'Nalldof should equal to len(b)'
        Nrow, Ncol = A.shape
        assert Nrow == len(b), 'Nrow should equal to len(b)'

        # --- --- #
        qf = GaussLegendreQuadrature(p + 3)  # the integral points on edges (1D)
        bcs, ws = qf.quadpts, qf.weights  # bcs.shape: (NQ,2); ws.shape: (NQ,)
        ps = np.einsum('ij, kjm->ikm', bcs, node[edge])  # ps.shape: (NQ,NE,2), NE is the number of edges

        vephi = self.vSpace.edge_basis(ps, index=None, p=p)
        # # (NQ,NE,eldof), eldof is the number of local 1D dofs on one edge
        invEM = self.vSpace.invEM  # (NE,eldof,eldof)

        # --- project uD to uDP on Dirichlet edges --- #
        uDI = uD(ps[:, idxDirEdge, :])  # (NQ,NE_Dir,2), get the Dirichlet values at physical integral points
        uDrhs = np.einsum('i, ijn, ijm, j->jmn', ws, uDI, vephi[:, idxDirEdge, :], hE[idxDirEdge])  # (NE_Dir,eldof,2)
        uDP = np.einsum('ijk, ikn->ijn', invEM[idxDirEdge, ...], uDrhs)  # (NE_Dir,eldof,eldof)x(NE_Dir,eldof,2)=>(NE_Dir,eldof,2)
        uDP0 = uDP[..., 0]
        uDP1 = uDP[..., 1]

        # --- apply to the left-matrix and right-vector --- #
        x = np.zeros((Nrow, 1), dtype=np.float)
        x[idxDirDof, 0] = np.concatenate([uDP0.flatten(), uDP1.flatten()])
        b -= A @ x
        bdIdx = np.zeros(Nrow, dtype=np.int)
        bdIdx[idxDirDof] = 1
        Tbd = spdiags(bdIdx, 0, Nrow, Nrow)
        T = spdiags(1 - bdIdx, 0, Nrow, Nrow)
        A = T @ A @ T + Tbd

        b[idxDirDof] = x[idxDirDof]
        return A, b

    def setStokesDirichletDofs(self, idxDirEdge=None, StaticCondensation=False):
        eldof = self.p + 1
        NC = self.mesh.number_of_cells()
        NE = self.mesh.number_of_edges()
        vldof = self.vSpace.smspace.number_of_local_dofs()
        vgdof = self.vSpace.number_of_global_dofs()
        pgdof = self.pSpace.number_of_global_dofs()

        idxDirEdge = self.defaultDirichletEdges() if idxDirEdge is None else idxDirEdge
        idxDirDof0 = eldof * idxDirEdge.reshape(-1, 1) + np.arange(eldof)
        idxDirDof0 = np.squeeze(idxDirDof0.reshape(1, -1))  # np.squeeze transform 2-D array (NDirDof,1) into 1-D (NDirDof,)

        if StaticCondensation is True:
            idxDirDof1 = NE*eldof + idxDirDof0
        else:
            Ncelldofs = NC * vldof
            idxDirDof0 += Ncelldofs
            idxDirDof1 = vgdof + idxDirDof0

        return np.concatenate([idxDirDof0, idxDirDof1])

    def defaultDirichletEdges(self):
        """
        This is the setting of default Dirichlet edges,
        in this function, all the edges are Dirichlet edges.
        :return:
        """
        mesh = self.mesh
        edge2cell = mesh.ds.edge_to_cell()
        isBdEdge = (edge2cell[:, 0] == edge2cell[:, 1])  # (NE,), the bool vars, to get the boundary edges

        isDirEdge = isBdEdge  # here, we set all the boundary edges are Dir edges
        idxDirEdge, = np.nonzero(isDirEdge)  # (NE_Dir,)
        return idxDirEdge

    def residual_estimate0(self, nu, uh, f=None, Dir_func=None, idxDirEdge=None):
        """
        :param nu: the nu in Stokes equation
        :param uh: uh = [uh0, uh1]
        :param f: the source vector
        :param Dir_func: the Dirichlet boundary true velocity vector
        :return: eta, the residual estimation
        """

        p = self.vSpace.p
        mesh = self.mesh
        GD = mesh.geo_dimension()
        vSpace = self.vSpace
        h = mesh.cell_area()  # (NC,)
        vdof = self.dof.velocityDof()  # note that, this is only the scalar variable's dof.
        grad_basis = self.vSpace.grad_basis  # grad_basis(point): (NQ,NC,pldof,2)
        integralalg = self.integralalg
        eldof = int(comb(p + GD - 1, GD - 1))
        ldof = int(comb(p + GD, GD))
        pldof = int(comb(p + 1 + GD, GD))
        node = mesh.entity('node')
        edge = mesh.entity('edge')
        NC = mesh.number_of_cells()
        NE = mesh.number_of_edges()

        if len(uh.shape) != GD:
            raise ValueError("len(uh.shape) != GD")

        if f is not None:
            fh = self.vSpace.project(f, dim=GD)
            # # (vgdof,2), the coefficients, fh[:,0] is the project of f1, and fh[:,1] is the project of f2
        else:
            fh = self.vSpace.project(self.vSpace.function(dim=GD), dim=GD)

        # --- begin the osc_f --- #
        def osc(x, index=np.s_[:]):
            fhval = vSpace.value(fh, x, index=index)  # fhval.shape: (NQ,NC,2)
            fval = f(x)
            return (fval - fhval)**2
        osc_f = np.sum(self.integralalg.integral(osc, celltype=True), axis=1)  # (NC,)
        osc_f = nu**(-1) * h**2 * osc_f  # (NC,)

        # --- begin the eta_s --- #
        cell2dof, cell2dofLocation = vdof.cell2dof, vdof.cell2dofLocation
        uh0 = uh[cell2dof, 0]
        uh0_c = np.split(uh0, cell2dofLocation[1:-1])  # list, each-term.shape: (Cldof,), split into each cell
        uh1 = uh[cell2dof, 1]
        uh1_c = np.split(uh1, cell2dofLocation[1:-1])  # list, each-term.shape: (Cldof,), split into each cell
        stabM = vSpace.reconstruction_stabilizer_matrix()  # list, (NC,): each-term's shape: (Cldof,Cldof)

        def get_eta_s(x):
            uh0c = x[0]  # (Cldof,)
            uh1c = x[1]  # (Cldof,)
            stabMc = x[2]  # (Cldof,Cldof)
            return uh0c@stabMc@uh0c + uh1c@stabMc@uh1c
        eta_s = nu * np.array(list(map(get_eta_s, zip(uh0_c, uh1_c, stabM))))  # array, (NC,)
        # # eta_s = nu * (uh.flatten('F') @ stabM @ uh.flatten('F'))

        # --- begin the eta_d_temp --- #
        RM = vSpace.RM  # (psmldof,\sum_C{Cldof}); RM is the final reconstruction matrix
        RM_c = np.hsplit(RM, cell2dofLocation[1:-1])  # list, each-term.shape: (psmldof,Cldof), split into each cell

        def gphixx(point, index):
            gphi = grad_basis(point, index=index, p=p + 1)  # gphi: (NQ,NC,pldof,2)
            return np.einsum('ijk, ijp->ijpk', gphi[..., 0], gphi[..., 0])
        gmat_xx = integralalg.integral(gphixx, celltype=True, q=p + 3)  # (NC,pldof,pldof)

        def gphixy(point, index):
            gphi = grad_basis(point, index=index, p=p + 1)  # gphi: (NQ,NC,pldof,2)
            return np.einsum('ijk, ijp->ijpk', gphi[..., 1], gphi[..., 0])
        gmat_xy = integralalg.integral(gphixy, celltype=True, q=p + 3)  # (NC,pldof,pldof)

        def gphiyy(point, index):
            gphi = grad_basis(point, index=index, p=p + 1)  # gphi: (NQ,NC,pldof,2)
            return np.einsum('ijk, ijp->ijpk', gphi[..., 1], gphi[..., 1])
        gmat_yy = integralalg.integral(gphiyy, celltype=True, q=p + 3)  # (NC,pldof,pldof)

        def div_square(x):
            uh0c = x[0]  # (Cldof,)
            uh1c = x[1]  # (Cldof,)
            rmc = x[2]  # (pldof,Cldof)
            gmatxx = x[3]  # (pldof,pldof)
            gmatxy = x[4]  # (pldof,pldof)
            gmatyy = x[5]  # (pldof,pldof)

            p0 = rmc@uh0c
            p1 = rmc@uh1c

            mxx = p0@gmatxx@p0
            mxy = p0@gmatxy@p1
            myy = p1@gmatyy@p1
            return mxx + 2*mxy + myy
        eta_d = nu * np.array(list(map(div_square, zip(uh0_c, uh1_c, RM_c, gmat_xx, gmat_xy, gmat_yy))))  # array, (NC,)

        # --- other div test --- #
        # actually, the following is only the divergence of uh, not the div of r_h^{k+1}\hhovector{u}_h.
        def grad_uh(point, index=np.s_[:]):
            return vSpace.grad_value(uh, point=point, index=index)
        guh = integralalg.integral(grad_uh, celltype=True)
        # # (NC,GD,GD): guh[...,0] is the \nabla_x of uh, and guh[...,1] is the \nabla_y

        uh1x = guh[:, 0, 0]  # (NC,)
        # uh1y = guh[:, 0, 1]  # (NC,)
        # uh2x = guh[:, 1, 0]  # (NC,)
        uh2y = guh[:, 1, 1]  # (NC,)
        eta_d_temp = nu * (uh1x*uh1x + 2*uh1x*uh2y + uh2y*uh2y)
        # --- other div test --- #

        # --- eta_J --- #
        edge2cell = mesh.ds.edge_to_cell()
        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
        inedgeIdx = np.nonzero(isInEdge)
        bdcellIdx = edge2cell[idxDirEdge, 0]

        def func_ruh(x):
            uh0c = x[0]
            uh1c = x[1]
            RMc = x[2]

            ruh0 = RMc@uh0c
            ruh1 = RMc@uh1c
            return np.concatenate([ruh0, ruh1])[np.newaxis, :]  # (1,2*pldof)
        ruh = np.concatenate(list(map(func_ruh, zip(uh0_c, uh1_c, RM_c))))  # (NC,2*pldof)
        ruh0 = ruh[:, :pldof]
        ruh1 = ruh[:, pldof:]

        # # get jumps on interior edges
        hE = integralalg.edgemeasure  # (NE,), the length of edges
        qf = GaussLegendreQuadrature(p + 3)
        bcs, ws = qf.quadpts, qf.weights  # bcs.shape: (NQ,2); ws.shape: (NQ,)
        ps = np.einsum('ij, kjm->ikm', bcs, node[edge])  # (NQ,NE,2), NE is the number of edges

        ephi = vSpace.edge_basis(ps)  # (NQ,NE,eldof), eldof is the number of local 1D dofs on one edge
        pphi_oneside = vSpace.basis(ps, index=edge2cell[:, 0], p=p + 1)  # (NQ,NE,pldof)
        pphi_otherside = vSpace.basis(ps[:, isInEdge, :], index=edge2cell[isInEdge, 1], p=p + 1)  # (NQ,NInE,pldof)

        evruh0_oneside = np.einsum('...ij, ...ij->...i', pphi_oneside, ruh0[edge2cell[:, 0], ...])  # (NQ,NE)
        evruh0_otherside = np.einsum('...ij, ...ij->...i', pphi_otherside, ruh0[edge2cell[isInEdge, 1], ...])  # (NQ,NE)
        evruh1_oneside = np.einsum('...ij, ...ij->...i', pphi_oneside, ruh1[edge2cell[:, 0], ...])  # (NQ,NInE)
        evruh1_otherside = np.einsum('...ij, ...ij->...i', pphi_otherside, ruh1[edge2cell[isInEdge, 1], ...])  # (NQ,NInE)

        # # jump at interior edges
        uh0_jump = np.zeros((NE,))
        jump0 = evruh0_oneside[..., isInEdge] - evruh0_otherside  # (NQ,NInE)
        uh0_jump[isInEdge] = hE[isInEdge]**(-1) * np.einsum('i,ij,j->j', ws, jump0*jump0, hE[isInEdge])

        uh1_jump = np.zeros((NE,))
        jump1 = evruh1_oneside[..., isInEdge] - evruh1_otherside  # (NQ,NInE)
        uh1_jump[isInEdge] = hE[isInEdge]**(-1) * np.einsum('i,ij,j->j', ws, jump1 * jump1, hE[isInEdge])

        # TODO: 检查 Neumann 边上的跳跃形式. 这里只针对 Dirichelt 边 !!!
        # # jump at Dirichlet boundary edges
        add_bdJump = True
        if add_bdJump:
            # --- one way ---- #
            # gh = self.vSpace.project(u, dim=GD)  # the aim is to get the projection of u on the boundary edges
            # gh_edge0 = gh[NC * ldof:, 0].reshape(NE, eldof)
            # gh_edge1 = gh[NC * ldof:, 1].reshape(NE, eldof)
            # evgh_bdedge0 = np.einsum('...ij,ij->...i', ephi[..., idxDirEdge, :], gh_edge0[idxDirEdge, ...])  # (NQ,NbdE)
            # evgh_bdedge1 = np.einsum('...ij,ij->...i', ephi[..., idxDirEdge, :], gh_edge1[idxDirEdge, ...])  # (NQ,NbdE)
            # jump0 = evgh_bdedge0 - evruh0_oneside[:, idxDirEdge]
            # jump1 = evgh_bdedge1 - evruh1_oneside[:, idxDirEdge]

            # --- another way --- #
            ev_u = Dir_func(ps)  # (NQ,NE,2)
            jump0 = ev_u[:, idxDirEdge, 0] - evruh0_oneside[:, idxDirEdge]
            jump1 = ev_u[:, idxDirEdge, 1] - evruh1_oneside[:, idxDirEdge]

            # --- add
            uh0_jump[idxDirEdge] = hE[idxDirEdge] ** (-1) * np.einsum('i,ij,j->j', ws, jump0 * jump0, hE[idxDirEdge])
            uh1_jump[idxDirEdge] = hE[idxDirEdge] ** (-1) * np.einsum('i,ij,j->j', ws, jump1 * jump1, hE[idxDirEdge])

        # # ---
        cell2edge, _ = mesh.ds.cell_to_edge()
        edgeLocation = np.add.accumulate(mesh.number_of_edges_of_cells())
        c2e_split = np.split(cell2edge, edgeLocation[:-1])

        def func_eta_J(c2e_c):
            return sum(uh0_jump[c2e_c] + uh1_jump[c2e_c])
        eta_J = nu * (np.array(list(map(func_eta_J, c2e_split))))

        # --- TODO: 当误差量级为 1e-8 左右时, eta_s 和 eta_d 会出现负数的情况.
        return np.sqrt(1.0*osc_f + 1.0*np.abs(eta_s) + 1.0*np.abs(eta_d) + 1.0*eta_J)

    def posterror_enengyerror(self, nu, grad_u, uh):
        p = self.vSpace.p
        mesh = self.mesh
        GD = mesh.geo_dimension()
        vSpace = self.vSpace
        h = mesh.cell_area()  # (NC,)
        vdof = self.dof.velocityDof()  # note that, this is only the scalar variable's dof.
        grad_basis = self.vSpace.grad_basis  # grad_basis(point): (NQ,NC,pldof,2)
        integralalg = self.integralalg
        eldof = int(comb(p + GD - 1, GD - 1))
        ldof = int(comb(p + GD, GD))
        pldof = int(comb(p + 1 + GD, GD))
        node = mesh.entity('node')
        edge = mesh.entity('edge')
        NC = mesh.number_of_cells()
        NE = mesh.number_of_edges()

        # --- stability term --- #
        cell2dof, cell2dofLocation = vdof.cell2dof, vdof.cell2dofLocation
        uh0 = uh[cell2dof, 0]
        uh0_c = np.split(uh0, cell2dofLocation[1:-1])  # list, each-term.shape: (Cldof,), split into each cell
        uh1 = uh[cell2dof, 1]
        uh1_c = np.split(uh1, cell2dofLocation[1:-1])  # list, each-term.shape: (Cldof,), split into each cell
        stabM_c = vSpace.reconstruction_stabilizer_matrix()  # list, (NC,): each-term's shape: (Cldof,Cldof)
        RM = vSpace.RM  # (psmldof,\sum_C{Cldof}); RM is the final reconstruction matrix
        RM_c = np.hsplit(RM, cell2dofLocation[1:-1])  # list, each-term.shape: (psmldof,Cldof), split into each cell

        # # stability error
        def get_posterr(x):
            uh0c = x[0]  # (Cldof,)
            uh1c = x[1]  # (Cldof,)
            stabMc = x[2]  # (Cldof,Cldof)
            RMc = x[3]  # (psmldof,Cldof)
            staberr = np.array([uh0c @ stabMc @ uh0c + uh1c @ stabMc @ uh1c])  # np.array, shape:(1,)

            ruh0 = RMc@uh0c  # (psmldof,1)
            ruh1 = RMc@uh1c
            return np.concatenate([staberr, ruh0, ruh1])

        postSomething = np.array(list(map(get_posterr, zip(uh0_c, uh1_c, stabM_c, RM_c))))  # array, (NC,2*psmldof+1)
        post_staberr = nu * postSomething[:, 0]  # (NC,1)
        ruh0 = postSomething[:, 1:1+pldof].reshape(-1, 1)  # (NC*psmldof,1)
        ruh1 = postSomething[:, 1+pldof:].reshape(-1, 1)  # (NC*psmldof,1)
        ruh = np.concatenate([ruh0, ruh1], axis=1)  # (NC*psmldof,2)

        # # grad error
        def get_garderr(point, index=np.s_[:]):
            val_grad_u = grad_u(point)
            u0x = val_grad_u[..., 0, 0]  # have the same shape with point
            u0y = val_grad_u[..., 0, 1]
            u1x = val_grad_u[..., 1, 0]
            u1y = val_grad_u[..., 1, 1]

            # # grad value, (NC,point.shape,GD,GD): guh[...,0] is the \nabla_x of uh, and guh[...,1] is the \nabla_y
            gruh = self.smspace_grad_value(ruh, point, index=index, p=p + 1)
            gruh0x = gruh[..., 0, 0]
            gruh0y = gruh[..., 0, 1]
            gruh1x = gruh[..., 1, 0]
            gruh1y = gruh[..., 1, 1]
            # gruh0 = self.smspace_grad_value(ruh0, point, index=index, p=p+1)
            # gruh1 = self.smspace_grad_value(ruh1, point, index=index, p=p + 1)
            # gruh0x = np.squeeze(gruh0[..., 0])
            # gruh0y = np.squeeze(gruh0[..., 1])
            # gruh1x = np.squeeze(gruh1[..., 0])
            # gruh1y = np.squeeze(gruh1[..., 1])

            return (u0x - gruh0x)**2 + (u0y - gruh0y)**2 + (u1x - gruh1x)**2 + (u1y - gruh1y)**2
        post_graderr = nu * integralalg.integral(get_garderr, celltype=True)  # (NC,)
        # post_graderr = 0

        # --- TODO: 同 residual_estimate0() 中的情况, 当误差量级为 1e-8 左右时, post_staberr 会出现负数的情况.
        return np.sqrt(post_graderr + np.abs(post_staberr))

    def smspace_grad_value(self, uh, point, index=np.s_[:], p=None):
        p = self.p if p is None else p
        smspace = self.vSpace.smspace
        gphi = smspace.grad_basis(point, index=index, p=p)
        cell2dof = smspace.dof.cell_to_dof(p=p)
        if (type(index) is np.ndarray) and (index.dtype.name == 'bool'):
            N = np.sum(index)
        elif type(index) is slice:
            N = len(cell2dof)
        else:
            N = len(index)
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        if point.shape[-2] == N:
            s1 = '...ijm, ij{}->...i{}m'.format(s0[:dim], s0[:dim])
            return np.einsum(s1, gphi, uh[cell2dof[index]])
        elif point.shape[0] == N:
            return np.einsum('ikjm, ij->ikm', gphi, uh[cell2dof[index]])

    def post_estimator_effective(self, eta=None):
        pass

    def post_estimator_markcell(self, eta, theta=0.3):
        etasqu = eta*eta
        sumeta = np.sum(etasqu)
        sortedIdx = np.argsort(etasqu)[::-1]  # 降序排序后返回索引值
        eta1 = np.sort(etasqu)[::-1]  # 降序排序
        acceta1 = np.add.accumulate(eta1)

        biggerValIdx = np.arange(len(eta))[acceta1 >= theta*sumeta]
        biggestValIdx = min(biggerValIdx)

        isMarked = sortedIdx[:(biggestValIdx+1)]
        isMarked = sortedIdx[0] if len(isMarked) == 0 else isMarked
        # cellstart = self.mesh.ds.cellstart if hasattr(self.mesh.ds, 'cellstart') else 0
        isMarkedCell = self.mesh.mark_helper(isMarked)

        return isMarkedCell
























