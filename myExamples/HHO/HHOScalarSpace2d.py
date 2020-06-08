#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: HHOScalarSpace2d.py
# @Author: Yongchao Zhang
# @E-mail: yoczhang@126.com
# @Time: Feb 14, 2020
# ---


import numpy as np
from numpy.linalg import inv
from fealpy.common import block
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye

from fealpy.functionspace.function import Function
from fealpy.quadrature import GaussLegendreQuadrature
from fealpy.quadrature import PolygonMeshIntegralAlg
# from fealpy.functionspace.ScaledMonomialSpace2d import SMDof2d, ScaledMonomialSpace2d
from myScaledMonomialSpace2d import SMDof2d, ScaledMonomialSpace2d


class HHODof2d(object):
    """
    The dof manager of HHO 2d space.
    """

    def __init__(self, mesh, p):
        self.p = p
        self.mesh = mesh
        self.cell2dof, self.cell2dofLocation = self.cell_to_dof()
        self.multiIndex1d = self.multi_index_matrix1d()

    def multi_index_matrix1d(self):
        p = self.p
        ldof = p + 1
        multiIndex = np.zeros((ldof, 2), dtype=np.int)
        multiIndex[:, 0] = np.arange(p, -1, -1)
        multiIndex[:, 1] = p - multiIndex[:, 0]
        return multiIndex

    def boundary_dof(self):
        gdof = self.number_of_global_dofs()
        isBdDof = np.zeros(gdof, dtype=np.bool)
        edge2dof = self.edge_to_dof()
        isBdEdge = self.mesh.ds.boundary_edge_flag()
        isBdDof[edge2dof[isBdEdge]] = True
        return isBdDof

    def edge_to_dof(self):
        p = self.p
        mesh = self.mesh
        NC = mesh.number_of_cells()
        cldof = (p + 1) * (p + 2) // 2
        NE = mesh.number_of_edges()
        edge2dof = NC*cldof + np.arange(NE * (p + 1)).reshape(NE, p + 1)
        return edge2dof

    def cell_to_dof(self):
        """
        Construct the cell2dof array which are 1D array with a location array
        cell2dofLocation.

        The following code give the dofs of i-th cell.

        In each cell, the cell-dofs is the first, then is the edge-dofs.

        cell2dof[cell2dofLocation[i]:cell2dofLocation[i+1]]
        """
        p = self.p
        mesh = self.mesh
        idof = (p + 1) * (p + 2) // 2
        eldof = p + 1
        # cellLocation = mesh.ds.cellLocation
        # cell2edge = mesh.ds.cell_to_edge(return_sparse=False)

        NC = mesh.number_of_cells()

        ldof = self.number_of_local_dofs()
        cell2dofLocation = np.zeros(NC + 1, dtype=np.int)
        cell2dofLocation[1:] = np.add.accumulate(ldof)
        cell2dof = np.zeros(cell2dofLocation[-1], dtype=np.int)

        edge2dof = self.edge_to_dof()
        edge2cell = mesh.ds.edge_to_cell()
        idx = cell2dofLocation[edge2cell[:, [0]]] + edge2cell[:, [2]] * eldof + np.arange(eldof) + idof
        cell2dof[idx] = edge2dof

        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
        idx = (cell2dofLocation[edge2cell[isInEdge, 1]] + edge2cell[isInEdge, 3] * eldof).reshape(-1, 1) + np.arange(
            eldof) + idof
        cell2dof[idx] = edge2dof[isInEdge]

        idx = cell2dofLocation[:-1].reshape(-1, 1) + np.arange(idof)  # (NC,smldof)
        cell2dof[idx] = np.arange(NC * idof).reshape(NC, idof)
        return cell2dof, cell2dofLocation

    def number_of_global_dofs(self):
        p = self.p
        mesh = self.mesh
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()
        gdof = NE * (p + 1) + NC * (p + 1) * (p + 2) // 2
        return gdof

    def number_of_local_dofs(self, p=None):
        p = self.p if p is None else p
        mesh = self.mesh
        NCE = mesh.number_of_edges_of_cells()
        # # if mesh is triangular or quadrilateral mesh, NCE will be the int value
        if isinstance(NCE, int):
            NC = mesh.number_of_cells()
            NCE = NCE*np.ones((NC,), dtype=int)
        ldofs = NCE * (p + 1) + (p + 1) * (p + 2) // 2
        return ldofs

    def number_of_cell_local_dof(self):
        """
        :return: Number of local dof in cell
        """
        p = self.p
        return (p + 1) * (p + 2) // 2


class HHOScalarSpace2d(object):
    def __init__(self, mesh, p, q=None):
        self.p = p
        self.mesh = mesh
        self.smspace = ScaledMonomialSpace2d(mesh, p, q=q)
        self.smldof = self.smspace.number_of_local_dofs()
        self.psmldof = self.smspace.number_of_local_dofs(p=p+1)

        self.cellsize = self.smspace.cellsize

        self.ftype = self.mesh.ftype
        self.itype = self.mesh.itype

        self.dof = HHODof2d(mesh, p)

        self.integralalg = self.smspace.integralalg

        self.CRM = self.cell_righthand_matrix()  # (psmldof,\sum_C{Cldof})

        self.RM = self.reconstruction_matrix()  # (psmldof,\sum_C{Cldof})

        self.CM = self.smspace.cell_mass_matrix()  # (NC,smldof,smldof), smldof is the number of local dofs of smspace
        self.EM = self.smspace.edge_mass_matrix()  # (NE,eldof,eldof), eldof is the number of local 1D dofs on one edge
        self.invCM = inv(self.CM)  # (NC,smldof,smldof)
        self.invEM = inv(self.EM)  # (NE,eldof,eldof)

    def number_of_local_dofs(self, p=None):
        return self.dof.number_of_local_dofs(p=p)

    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def boundary_dof(self):
        return self.dof.boundary_dof()

    def edge_to_dof(self):
        return self.dof.edge_to_dof()

    def cell_to_dof(self, doftype='all'):
        if doftype == 'all':
            return self.dof.cell2dof, self.dof.cell2dofLocation
        elif doftype == 'cell':
            p = self.p
            NE = self.mesh.number_of_edges()
            NC = self.mesh.number_of_cells()
            idof = (p + 1) * (p + 2) // 2
            cell2dof = NE * (p + 1) + np.arange(NC * idof).reshape(NC, idof)
            return cell2dof

    def cell_righthand_matrix(self):
        p = self.p
        mesh = self.mesh
        NC = mesh.number_of_cells()
        node = mesh.entity('node')
        edge = mesh.entity('edge')
        edge2cell = mesh.ds.edge_to_cell()
        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])

        hE = self.integralalg.edgemeasure  # (NE,), the length of edges
        n = mesh.edge_unit_normal()  # (NE,2), the unit normal vector of edges
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
        phi0 = self.basis(ps, index=edge2cell[:, 0])  # (NQ,NE,smldof)
        phi1 = self.basis(ps[:, isInEdge, :], index=edge2cell[isInEdge, 1])

        gpphi0 = self.grad_basis(ps, index=edge2cell[:, 0], p=p + 1)  # (NQ,NE,psmldof,2),
        gpphi1 = self.grad_basis(ps[:, isInEdge, :], index=edge2cell[isInEdge, 1], p=p + 1)  # (NQ,NInE,psmldof,2)

        ephi = self.edge_basis(ps)  # (NQ,NE,eldof), eldof is the number of local 1D dofs on one edge

        # --- construct different matrix --- #
        smldof = self.smldof
        psmldof = self.psmldof
        eldof = p + 1  # the number of local 1D dofs on one edge
        cell2dof, cell2dofLocation = self.dof.cell2dof, self.dof.cell2dofLocation  # cell2dof.shape: (\sum_C{Cldof},)
        CRM = np.zeros((psmldof, cell2dofLocation[-1]), dtype=np.float)  # (psmldof,\sum_C{Cldof}), Cldof is the number of dofs in one cell

        # --- edge integration. Part I: (-v_T, \nabla w\cdot n)_{\partial T}
        T0 = np.einsum('i, ijk, ijmn, jn, j->jmk', ws, phi0, gpphi0, n, hE)  # (NE,psmldof,smldof)
        T1 = np.einsum('i, ijk, ijmn, jn, j->jmk', ws, phi1, gpphi1,
                       -n[isInEdge, :], hE[isInEdge])  # (NInE,psmldof,smldof)
        T = np.zeros((NC, psmldof, smldof), dtype=np.float)  # (NC,psmldof,smldof)
        np.add.at(T, edge2cell[:, 0], -T0)
        np.add.at(T, edge2cell[isInEdge, 1], -T1)

        # --- edge integration. Part II: (v_F, \nabla w\cdot n)_{\partial T}
        F0 = np.einsum('i, ijk, ijmn, jn, j->mjk', ws, ephi, gpphi0, n, hE)  # (psmldof,NE,eldof)
        F1 = np.einsum('i, ijk, ijmn, jn, j->mjk', ws, ephi[:, isInEdge, :], gpphi1,
                       -n[isInEdge, :], hE[isInEdge])  # (psmldof,NInE,eldof)

        idx = cell2dofLocation[edge2cell[:, [0]]] + edge2cell[:, [2]] * eldof + np.arange(eldof)  # (NE,eldof)
        idx += smldof  # rearrange the dofs
        CRM[:, idx] = F0  # (psmldof,\sum_C{Cldof}), Cldof is the number of dofs in one cell
        idx = cell2dofLocation[edge2cell[isInEdge, 1]].reshape(-1, 1) + \
              eldof * edge2cell[isInEdge, 3].reshape(-1, 1) + np.arange(eldof)  # (NInE,eldof)
        idx += smldof  # rearrange the dofs
        CRM[:, idx] = F1

        # --- the stiff matrix, (\nabla v, \nabla w)_T
        def f(x, index=None):
            gphi = self.grad_basis(x, index=index)  # using the cell-integration, so gphi: (NQ,NC,ldof,2)
            gpphi = self.grad_basis(x, index=index, p=p + 1)  # using the cell-integration, so gpphi: (NQ,NC,lpdof,2)
            return np.einsum('...mn, ...kn->...km', gphi, gpphi)

        S = self.integralalg.integral(f, celltype=True)  # (NC,psmldof,smldof)
        np.add.at(T, np.arange(NC), S)  # T.shape: (NC,psmldof,smldof)
        idx = cell2dofLocation[0:-1].reshape(-1, 1) + np.arange(smldof)  # (NC,smldof)
        CRM[:, idx] = T.swapaxes(0, 1)  # CRM.shape: (psmldof,\sum_C{Cldof})

        return CRM

    def reconstruction_matrix(self):
        p = self.p
        CRM = self.CRM
        cell2dofLocation = self.dof.cell2dofLocation
        smldof = self.smldof

        # --- left stiff matrix and the additional condition --- #
        # def f(x, index):
        #     gpphi = self.grad_basis(x, index=index, p=p + 1)
        #     return np.einsum('...mn, ...kn->...km', gpphi, gpphi)
        # ls = self.integralalg.integral(f, celltype=True)  # (NC,psmldof,psmldof)
        ls = self.monomial_stiff_matrix(p=p+1)  # (NC,psmldof,psmldof)

        def f(x, index):
            return self.basis(x, index=index, p=p + 1)
        l1 = self.integralalg.integral(f, celltype=True)  # (NC,psmldof)

        def f(x, index):
            return self.basis(x, index=index, p=p)
        r1 = self.integralalg.integral(f, celltype=True)  # (NC,smldof)

        # --- modify the matrix --- #
        ls[:, 0, :] = l1
        idx = cell2dofLocation[0:-1].reshape(-1, 1) + np.arange(smldof)  # (NC,smldof)
        CRM[0, idx] = r1  # (NC,\sum_C{Cldof}), Cldof is the number of dofs in one cell

        # --- reconstruction matrix --- #
        invls = inv(ls)  # (NC,psmldof,psmldof)
        Csplit = np.hsplit(CRM, cell2dofLocation[1:-1])  # list, len(Csplit) is NC, Csplit[i].shape is (psmldof,Cldof)

        f = lambda x: x[0] @ x[1]
        RM = np.concatenate(list(map(f, zip(invls, Csplit))), axis=1)  # (psmldof,\sum_C{Cldof})

        return RM  # (psmldof,\sum_C{Cldof})

    def reconstruction_stiff_matrix(self):
        p = self.p
        RM = self.RM  # (psmldof,\sum_C{Cldof}), Cldof is the number of dofs in one cell
        Sp = self.monomial_stiff_matrix(p=p+1)  # (NC,psmldof,psmldof)
        # sp1 = np.squeeze((self.stiff_matrix(p=p+1)).todense())

        cell2dofLocation = self.dof.cell2dofLocation
        Rsplit = np.hsplit(RM, cell2dofLocation[1:-1])  # list, len(Rsplit) is NC, Rsplit[i].shape is (psmldof,Cldof)

        def f(x): return np.transpose(x[1]) @ x[0] @ x[1]
        StiffM = list(map(f, zip(Sp, Rsplit)))  # list, its len is NC, each-term.shape: (Cldof,Cldof)

        return StiffM

    def projection_psmspace_to_smspace(self):
        p = self.p

        def rf(x, index):
            pphi = self.basis(x, index=index, p=p+1)  # (NQ,...,psmldof)
            phi = self.basis(x, index=index, p=p)  # (NQ,...,smldof)
            return np.einsum('...m, ...n->...nm', pphi, phi)
        rm = self.integralalg.integral(rf, celltype=True)  # (NC,smldof,psmldof)

        invCM = self.invCM

        return invCM@rm  # (NC,smldof,psmldof)

    def projection_sm_psm_space_to_edge(self):
        """
        projection from smspace, psmsace to edge 1d space
        """
        p = self.p
        mesh = self.mesh
        NC = mesh.number_of_cells()
        NE = mesh.number_of_edges()
        node = mesh.entity('node')
        edge = mesh.entity('edge')
        edge2cell = mesh.ds.edge_to_cell()
        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])

        hE = self.integralalg.edgemeasure  # (NE,), the length of edges
        n = mesh.edge_unit_normal()  # (NE,2), the unit normal vector of edges
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
        phi0 = self.basis(ps, index=edge2cell[:, 0], p=p)  # (NQ,NE,psmldof)
        phi1 = self.basis(ps[:, isInEdge, :], index=edge2cell[isInEdge, 1], p=p)  # (NQ,NInE,psmldof)
        pphi0 = self.basis(ps, index=edge2cell[:, 0], p=p + 1)  # (NQ,NE,psmldof)
        pphi1 = self.basis(ps[:, isInEdge, :], index=edge2cell[isInEdge, 1], p=p + 1)  # (NQ,NInE,psmldof)

        ephi = self.edge_basis(ps)  # (NQ,NE,eldof), eldof is the number of local 1D dofs on one edge

        # --- construct different matrix --- #
        smldof = self.smldof
        psmldof = self.psmldof
        eldof = p + 1  # the number of local 1D dofs on one edge

        # --- edge integration --- #
        invEM = self.invEM  # (NE,eldof,eldof)
        F0 = invEM@np.einsum('i, ijk, ijm, j->jmk', ws, phi0, ephi, hE)  # (NE,eldof,smldof)
        F1 = invEM[isInEdge, ...]@np.einsum('i, ijk, ijm, j->jmk', ws, phi1, ephi[:, isInEdge, :], hE[isInEdge])  # (NInE,eldof,smldof)
        pF0 = invEM@np.einsum('i, ijk, ijm, j->jmk', ws, pphi0, ephi, hE)  # (NE,eldof,psmldof)
        pF1 = invEM[isInEdge, ...]@np.einsum('i, ijk, ijm, j->jmk', ws, pphi1, ephi[:, isInEdge, :], hE[isInEdge])  # (NInE,eldof,psmldof)

        NCE = mesh.number_of_edges_of_cells()
        if isinstance(NCE, int):
            NC = mesh.number_of_cells()
            NCE = NCE*np.ones((NC,), dtype=int)
        sumNCE = NCE.sum()  # sumNCE is sum of (the number of edges of each cell)
        sm2E = np.zeros((eldof, sumNCE*smldof), dtype=np.float)
        psm2E = np.zeros((eldof, sumNCE*psmldof), dtype=np.float)

        # # celledgedofs = self.number_of_local_dofs() - smldof
        celledgedofs = NCE * smldof
        cell2dofLocation = np.zeros(NC + 1, dtype=np.int)
        cell2dofLocation[1:] = np.add.accumulate(celledgedofs)

        # celledgedofs = self.number_of_local_dofs(p=p+1) - psmldof
        celledgedofs = NCE * psmldof
        pcell2dofLocation = np.zeros(NC + 1, dtype=np.int)
        pcell2dofLocation[1:] = np.add.accumulate(celledgedofs)

        # --- add to corresponding cells --- #
        idx = cell2dofLocation[edge2cell[:, [0]]] + edge2cell[:, [2]] * smldof + np.arange(smldof)  # (NE,smldof)
        sm2E[:, idx] = F0.swapaxes(0, 1)
        idx = pcell2dofLocation[edge2cell[:, [0]]] + edge2cell[:, [2]] * psmldof + np.arange(psmldof)  # (NE,psmldof)
        psm2E[:, idx] = pF0.swapaxes(0, 1)

        idx = cell2dofLocation[edge2cell[isInEdge, 1]].reshape(-1, 1) + \
              smldof * edge2cell[isInEdge, [3]].reshape(-1, 1) + np.arange(smldof)  # (NInE,smldof)
        sm2E[:, idx] = F1.swapaxes(0, 1)
        idx = pcell2dofLocation[edge2cell[isInEdge, 1]].reshape(-1, 1) + \
              psmldof * edge2cell[isInEdge, [3]].reshape(-1, 1) + np.arange(psmldof)  # (NInE,smldof)
        psm2E[:, idx] = pF1.swapaxes(0, 1)

        # # sm2E.shape: (eldof, sumNCE*smldof),    psm2E.shape: (eldof, sumNCE*psmldof)
        return sm2E, psm2E

    def reconstruction_stabilizer_matrix(self):
        p = self.p
        smldof = self.smldof
        psmldof = self.psmldof
        eldof = p + 1  # the number of local 1D dofs on one edge
        EM = self.EM  # EM is the smspace.edge_mass_matrix(), (NE,eldof,eldof)
        h = self.mesh.edge_length()

        # # reconstruction matrix
        RM = self.RM  # (psmldof,\sum_C{Cldof}), Cldof is the number of dofs in one cell
        cell2dofLocation = self.dof.cell2dofLocation
        Rsplit = np.hsplit(RM, cell2dofLocation[1:-1])  # list, len(Rsplit) is NC, each-term.shape is (psmldof,Cldof)

        # # projection psmldof_to_smldof
        psm2sm = self.projection_psmspace_to_smspace()  # (NC,smldof,psmldof)

        sm2edge, psm2edge = self.projection_sm_psm_space_to_edge()
        # # sm2edge.shape: (eldof, sumNCE*smldof),    psm2edge.shape: (eldof, sumNCE*psmldof),
        # # sumNCE is sum of (the number of edges of each cell)

        NCE = self.mesh.number_of_edges_of_cells()  # (NC,)
        if isinstance(NCE, int):
            NC = self.mesh.number_of_cells()
            NCE = NCE*np.ones((NC,), dtype=int)
        NCEacc = np.add.accumulate(NCE)
        sm2edgeS = np.hsplit(sm2edge, NCEacc[:-1]*smldof)
        # # list, its length is NC, each-term.shape: (eldof,NE_C*smldof), NE_C is the number of edges in one cell
        psm2edgeS = np.hsplit(psm2edge, NCEacc[:-1]*psmldof)

        def f(x):
            # # x[0].shape: (eldof,NE_C*psmldof), NE_C is the number of edges in one cell
            # # x[1].shape: (eldof,NE_C*smldof)
            # # x[2].shape: (smldof,psmldof)
            # # x[3].shape: (1,)

            # --- one way --- #
            # idx = smldof*np.arange(x[3])  # (NE_C,)
            # t1 = np.tile(x[2], (x[3], 1))  # (NE_C*smldof,psmldof)
            # t2 = np.einsum('ij, jk->ijk', x[1], t1)  # (eldof,NE_C*smldof,psmldof)
            # t2 = np.concatenate(np.split(t2, idx[1:], axis=1), axis=2)  # (eldof,smldof,NE_C*psmldof)
            # t2 = t2.sum(axis=1)  # (eldof,smldof,NE_C*psmldof) => (eldof,NE_C*psmldof)
            # return x[0] - t2  # (eldof,NE_C*psmldof)

            # --- another way --- #
            # # construct the block matrix
            l = [([0] * x[3]) for i in range(x[3])]
            for i in range(x[3]):
                l[i][i] = x[2]
            # # compute the product of the matrix
            t = x[1] @ block(l)  # (eldof,NE_C*psmldof)
            return x[0] - t  # (eldof,NE_C*psmldof)

        # # get the projection to edge matrix,
        # # the aim is to compute the matrix: J_{1m}^{-1}*H_{3_m} - J_{1m}^{-1}*H_{2_m}*G_{4}^{-1}*G_5
        P2E = list(map(f, zip(psm2edgeS, sm2edgeS, psm2sm, list(NCE))))  # list, its len is NC, each-term.shape: (eldof, NCE*psmldof)

        # --- construct the reconstruction-stabilizer matrix
        cell2edge = self.mesh.ds.cell_to_edge()
        NCEacc = np.concatenate([[0], NCEacc])
        Cidx = np.arange(len(NCE))

        def f(x):
            # # x[3] is the number of edges in this cell, i.e., NCE
            # # x[4] is the current cell index
            l = np.arange(x[3])*smldof  # (NCE*smldof,)
            t = np.concatenate(np.hsplit(x[0], l[1:]), axis=0)  # (NCE*eldof, smldof), x[0] is the sm2edge_in_one_cell
            t = np.concatenate([t, -np.eye(eldof*x[3])], axis=1)  # (NCE*eldof, Cldof), Cldof = smldof + NCE*eldof, the eye()-matrix needs to add minus
            l = np.arange(x[3])*eldof
            t = np.concatenate(np.vsplit(t, l[1:]), axis=1)  # (eldof, NCE*Cldof)

            l = [([0] * x[3]) for i in range(x[3])]
            for i in range(x[3]):
                l[i][i] = x[2]  # x[2] is the RM in one cell, x[2].shape: (psmldof, Cldof)
            t = t + x[1]@block(l)
            # # x[1] is the P2E in one cell, x[1].shape: (eldof, NCE*psmldof)
            # # block(l).shape: (NCE*psmldof, NCE*Cldof)
            # # t.shape: (eldof, NCE*Cldof)

            l = np.arange(x[3])*(smldof+eldof*x[3])
            tsplit = np.hsplit(t, l[1:])  # list, its len is NCE, each-term.shape: (eldof,Cldof)

            eidx = cell2edge.flatten()[NCEacc[x[4]]:NCEacc[x[4]+1]]
            CEM = EM[eidx, ...]  # (NCE,eldof,eldof)
            CEh = h[eidx]

            def f2(y): return 1./y[2]*(np.transpose(y[1]) @ y[0] @ y[1])  # each-time, result.shape: (Cldof,Clodf)
            sm = np.sum(list(map(f2, zip(CEM, tsplit, CEh))), axis=0)  # (Cldof,Cldof)
            return sm
        StabM = list(map(f, zip(sm2edgeS, P2E, Rsplit, list(NCE), list(Cidx))))
        # # list, its len is NC, each-term.shape: (Cldof,Cldof)
        return StabM

    def source_vector(self, f):
        phi = self.basis  # basis is inherited from class ScaledMonomialSpace2d()

        def u(x, index):
            return np.einsum('ij, ijm->ijm', f(x), phi(x, index=index))
            # # f(x).shape: (NQ,NC).    phi(x,...).shape: (NQ,NC,ldof)

        fh = self.integralalg.integral(u, celltype=True)  # (NC,ldof)
        # # integralalg.integral() is inherited from class ScaledMonomialSpace2d()

        return fh  # (NC,ldof)

    def monomial_stiff_matrix(self, p=None):
        """
        Get the stiff matrix on ScaledMonomialSpace2d.

        ---
        The mass matrix on ScaledMonomialSpace2d can be found in class ScaledMonomialSpace2d(): mass_matrix()

        """
        p = self.p if p is None else p

        # assert p >= 1, 'the polynomial-order should have p >= 1 '

        mesh = self.mesh
        node = mesh.entity('node')
        smdof = SMDof2d(mesh, p)

        NC = mesh.number_of_cells()
        edge = mesh.entity('edge')
        edge2cell = mesh.ds.edge_to_cell()
        nm = mesh.edge_normal()
        # # (NE,2). The length of the normal-vector isn't 1, is the length of corresponding edge.

        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])  # the bool vars, to get the inner edges

        qf = GaussLegendreQuadrature(p + 3)  # the integral points on edges (1D)
        bcs, ws = qf.quadpts, qf.weights  # bcs.shape: (NQ,2); ws.shape: (NQ,)
        ps = np.einsum('ij, kjm->ikm', bcs, node[edge])  # ps.shape: (NQ,NE,2), NE is the number of edges

        gphi0 = self.grad_basis(ps, index=edge2cell[:, 0], p=p)
        # # gphi0.shape: (NQ,NInE,ldof,2), NInE is the number of interior edges, lodf is the number of local DOFs
        # # gphi0 is the grad-value of the cell basis functions on the one-side of the corresponding edges.
        gphi1 = self.grad_basis(ps[:, isInEdge, :], index=edge2cell[isInEdge, 1], p=p)

        # # using the divergence-theorem to get the
        S0 = np.einsum('i, ijkm, ijpm->jpk', ws, gphi0, gphi0)  # (NE,ldof,ldof)
        b = node[edge[:, 0]] - self.smspace.cellbarycenter[edge2cell[:, 0]]  # (NE,2)
        S0 = np.einsum('ij, ij, ikm->ikm', b, nm, S0)  # (NE,ldof,ldof)

        S1 = np.einsum('i, ijkm, ijpm->jpk', ws, gphi1, gphi1)  # (NInE,ldof,ldof)
        b = node[edge[isInEdge, 0]] - self.smspace.cellbarycenter[edge2cell[isInEdge, 1]]  # (NInE,2)
        S1 = np.einsum('ij, ij, ikm->ikm', b, -nm[isInEdge], S1)  # (NInE,ldof,ldof)

        ldof = self.smspace.number_of_local_dofs(p=p)
        S = np.zeros((NC, ldof, ldof), dtype=np.float)
        np.add.at(S, edge2cell[:, 0], S0)
        np.add.at(S, edge2cell[isInEdge, 1], S1)

        multiIndex = smdof.multiIndex
        q = np.sum(multiIndex, axis=1) - 1  # here, we used the grad-basis to get stiff-matrix, so we need to -1
        qq = q + q.reshape(-1, 1) + 2
        qq[0, 0] = 1
        # # note that this is the special case, since we want to compute the \int_T \nabla u\cdot \nabla v,
        # # this needs to minus 1 in the 'q', so qq[0,0] is 0, moreover, S[:, 0, :] == S[:, :, 0] is 0-values,
        # # so we set qq[0, 0] = 1 which doesn't affect the result of S /= qq.

        S /= qq  # (NC,ldof,ldof)
        return S

    def basis(self, point, index=None, p=None):
        return self.smspace.basis(point, index=index, p=p)

    def grad_basis(self, point, index=None, p=None):
        return self.smspace.grad_basis(point, index=index, p=p)

    def value(self, uh, point, index=None):
        NC = self.mesh.number_of_cells()
        smldof = self.smldof
        return self.smspace.value(uh[:NC*smldof, ...], point, index=index)

    def grad_value(self, uh, point, index=None):
        NC = self.mesh.number_of_cells()
        smldof = self.smldof
        return self.smspace.grad_value(uh[:NC*smldof, ...], point, index=index)

    # def edge_basis(self, point, index=None, p=None):
    #     p = self.p if p is None else p
    #     index = index if index is not None else np.s_[:]
    #     center = self.integralalg.edgebarycenter
    #     h = self.integralalg.edgemeasure
    #     t = self.mesh.edge_unit_tagent()
    #     val = np.sum((point - center[index]) * t[index], axis=-1) / h[index]
    #     phi = np.ones(val.shape + (p + 1,), dtype=self.ftype)
    #     if p == 1:
    #         phi[..., 1] = -val
    #     else:
    #         phi[..., 1:] = -val[..., np.newaxis]
    #         np.multiply.accumulate(phi, axis=-1, out=phi)
    #     return phi
    def edge_basis(self, point, index=None, p=None):
        return self.smspace.edge_basis(point, index=index, p=p)

    def edge_value(self, uh, bcs):
        phi = self.edge_basis(bcs)
        edge2dof = self.dof.edge_to_dof()

        dim = len(uh.shape) - 1
        s0 = 'abcdefg'[:dim]
        s1 = '...ij, ij{}->...i{}'.format(s0, s0)
        val = np.einsum(s1, phi, uh[edge2dof])
        return val

    def project(self, u, dim=1):
        p = self.p
        mesh = self.mesh
        node = mesh.entity('node')
        edge = mesh.entity('edge')
        h = mesh.entity_measure('edge')
        NC = mesh.number_of_cells()
        smldof = self.smldof

        uh = self.function(dim=dim)

        qf = GaussLegendreQuadrature(p + 3)
        bcs, ws = qf.quadpts, qf.weights
        ps = np.einsum('ij, kjm->ikm', bcs, node[edge])
        uI = u(ps)

        ephi = self.edge_basis(ps)
        b = np.einsum('i, ij..., ijk, j->jk...', ws, uI, ephi, h)
        if dim == 1:
            uh[NC*smldof:, ...].flat = (self.invEM@b[:, :, np.newaxis]).flat
        else:
            uh[NC*smldof:, ...].flat = (self.invEM@b).flat

        t = 'd'
        s = '...{}, ...m->...m{}'.format(t[:dim > 1], t[:dim > 1])

        def f1(x, index):
            phi = self.basis(x, index)
            return np.einsum(s, u(x), phi)
        b = self.integralalg.integral(f1, celltype=True)
        if dim == 1:
            uh[:NC*smldof, ...].flat = (self.invCM@b[:, :, np.newaxis]).flat
        else:
            uh[:NC*smldof, ...].flat = (self.invCM@b).flat
        return uh  # uh[0:NC*smldof] is the dofs project on cells, uh[NC*smldof:end] is the dofs project on edges.

    def function(self, dim=None, array=None):
        f = Function(self, dim=dim, array=array)
        return f

    def array(self, dim=None):
        gdof = self.number_of_global_dofs()
        # gdof = len(self.dof.cell2dof)
        if dim in {None, 1}:
            shape = gdof
        elif type(dim) is int:
            shape = (gdof, dim)
        elif type(dim) is tuple:
            shape = (gdof, ) + dim
        return np.zeros(shape, dtype=np.float)

    def system_matrix(self):
        gdof = self.dof.number_of_global_dofs()
        StiffM = self.reconstruction_stiff_matrix()  # list, its len is NC, each-term.shape (Cldof,Cldof)
        StabM = self.reconstruction_stabilizer_matrix()  # list, its len is NC, each-term.shape (Cldof,Cldof)

        cell2dof, doflocation = self.dof.cell_to_dof()
        cell2dof_split = np.hsplit(cell2dof, doflocation[1:-1])

        def get_system_matrix(x):
            # # x[0], the stiff matrix in current cell
            # # x[1], the stab matrix in current cell
            # # x[2], the dofs-index in current cell
            StiffM_C = x[0]  # the left matrix at this cell
            StabM_C = x[1]
            dof_C = x[2]  # (NCdof,)
            Ndof_C = len(dof_C)

            # --- get the row and col index --- #
            rowIndex = np.einsum('i, k->ik', dof_C, np.ones(Ndof_C, ))
            colIndex = np.transpose(rowIndex)

            # --- add to the global matrix and vector --- #
            r = csr_matrix(((StiffM_C+StabM_C).flat, (rowIndex.flat, colIndex.flat)), shape=(gdof, gdof), dtype=np.float)
            return r
        M = sum(list(map(get_system_matrix, zip(StiffM, StabM, cell2dof_split))))
        return M

    def system_source(self, f):
        gdof = self.dof.number_of_global_dofs()
        fh = self.source_vector(f)  # (NC,ldof)
        shape = fh.shape
        V = np.zeros((gdof, 1), dtype=np.float)
        V[:(shape[0]*shape[1]), 0] = fh.flatten()
        return V

    def L2_error(self, uI, uh):
        eu = uI - uh

        def f(x, index):
            evalue = self.value(eu, x, index=index)  # the evalue has the same shape of x.
            return evalue*evalue
        err = self.integralalg.integral(f)
        return np.sqrt(err)

    def H1_semi_error(self, uI, uh):
        eu = uI - uh

        def f(x, index):
            evalue = self.grad_value(eu, x, index=index)  # the evalue has the same shape of x.
            return np.einsum('...n, ...n->...', evalue, evalue)
        err = self.integralalg.integral(f)
        return np.sqrt(err)




