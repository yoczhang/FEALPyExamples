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
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye

from fealpy.functionspace.function import Function
from fealpy.quadrature import GaussLegendreQuadrature
from fealpy.quadrature import PolygonMeshIntegralAlg
from fealpy.functionspace.ScaledMonomialSpace2d import ScaledMonomialSpace2d


class HHODof2d():
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
        NE = mesh.number_of_edges()
        edge2dof = np.arange(NE * (p + 1)).reshape(NE, p + 1)
        return edge2dof

    def cell_to_dof(self):
        """
        Construct the cell2dof array which are 1D array with a location array
        cell2dofLocation.

        The following code give the dofs of i-th cell.

        cell2dof[cell2dofLocation[i]:cell2dofLocation[i+1]]
        """
        p = self.p
        mesh = self.mesh
        cellLocation = mesh.ds.cellLocation
        cell2edge = mesh.ds.cell_to_edge(sparse=False)

        NC = mesh.number_of_cells()

        ldof = self.number_of_local_dofs()
        cell2dofLocation = np.zeros(NC + 1, dtype=np.int)
        cell2dofLocation[1:] = np.add.accumulate(ldof)
        cell2dof = np.zeros(cell2dofLocation[-1], dtype=np.int)

        edge2dof = self.edge_to_dof()
        edge2cell = mesh.ds.edge_to_cell()
        idx = cell2dofLocation[edge2cell[:, [0]]] + edge2cell[:, [2]] * (p + 1) + np.arange(p + 1)
        cell2dof[idx] = edge2dof

        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
        idx = (cell2dofLocation[edge2cell[isInEdge, 1]] + edge2cell[isInEdge, 3] * (p + 1)).reshape(-1, 1) + np.arange(
            p + 1)
        cell2dof[idx] = edge2dof[isInEdge]

        NV = mesh.number_of_vertices_of_cells()
        NE = mesh.number_of_edges()
        idof = (p + 1) * (p + 2) // 2
        idx = (cell2dofLocation[:-1] + NV * (p + 1)).reshape(-1, 1) + np.arange(idof)
        cell2dof[idx] = NE * (p + 1) + np.arange(NC * idof).reshape(NC, idof)
        return cell2dof, cell2dofLocation

    def number_of_global_dofs(self):
        p = self.p
        mesh = self.mesh
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()
        gdof = NE * (p + 1) + NC * (p + 1) * (p + 2) // 2
        return gdof

    def number_of_local_dofs(self):
        p = self.p
        mesh = self.mesh
        NCE = mesh.number_of_edges_of_cells()
        ldofs = NCE * (p + 1) + (p + 1) * (p + 2) // 2
        return ldofs


class HHOScalarSpace2d():
    def __init__(self, mesh, p, q=None):
        self.p = p
        self.mesh = mesh
        self.smspace = ScaledMonomialSpace2d(mesh, p, q=q)

        self.cellsize = self.smspace.cellsize

        self.ftype = self.mesh.ftype
        self.itype = self.mesh.itype

        self.dof = HHODof2d(mesh, p)

        self.integralalg = self.smspace.integralalg

        self.Co = self.construction_matrix()  # (psmldof,NC*Cldof)

        self.CM = self.smspace.cell_mass_matrix()  # (NC,smldof,smldof), smldof is the number of local dofs of smspace
        self.EM = self.smspace.edge_mass_matrix()  # (NE,eldof,eldof), eldof is the number of local 1D dofs on one edge

    def number_of_local_dofs(self):
        return self.dof.number_of_local_dofs()

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

    def construction_matrix(self):
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
        smldof = self.smspace.number_of_local_dofs()
        psmldof = self.smspace.number_of_local_dofs(p=p + 1)
        eldof = p + 1  # the number of local 1D dofs on one edge
        cell2dof, cell2dofLocation = self.dof.cell2dof, self.dof.cell2dofLocation
        Co = np.zeros((psmldof, len(cell2dof)),
                     dtype=np.float)  # (psmldof,NC*Cldof), Cldof is the number of dofs in one cell

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
        Co[:, idx] = F0
        idx = cell2dofLocation[edge2cell[isInEdge, 1]].reshape(-1, 1) + \
              eldof * edge2cell[isInEdge, [3]].reshape(-1, 1) + np.arange(eldof)  # (NInE,eldof)
        idx += smldof  # rearrange the dofs
        Co[:, idx] = F1

        # --- the stiff matrix, (\nabla v, \nabla w)_T
        def f(x, index):
            gphi = self.grad_basis(x, index=index)
            gpphi = self.grad_basis(x, index=index, p=p + 1)
            return np.einsum('...mn, ...kn->...km', gphi, gpphi)

        S = self.integralalg.integral(f, celltype=True)  # (NC,psmldof,smldof)
        np.add.at(T, np.arange(NC), S)  # T.shape: (NC,psmldof,smldof)
        idx = cell2dofLocation[0:-1].reshape(-1, 1) + np.arange(smldof)  # (NC,smldof)
        Co[:, idx] = T.swapaxes(0, 1)  # R.shape: (psmldof,NC*Cldof)

        return Co

    def reconstruction_matrix(self):
        p = self.p
        Co = self.Co
        cell2dofLocation = self.dof.cell2dofLocation
        smldof = self.smspace.number_of_local_dofs()

        # --- left stiff matrix and the additional condition
        def f(x, index):
            gpphi = self.grad_basis(x, index=index, p=p + 1)
            return np.einsum('...mn, ...kn->...km', gpphi, gpphi)
        ls = self.integralalg.integral(f, celltype=True)  # (NC,psmldof,psmldof)

        def f(x, index):
            return self.basis(x, index=index, p=p + 1)
        l1 = self.integralalg.integral(f, celltype=True)  # (NC,psmldof)

        def f(x, index):
            return self.basis(x, index=index, p=p)
        r1 = self.integralalg.integral(f, celltype=True)  # (NC,smldof)

        # --- modify the matrix
        ls[:, 0, :] = l1
        idx = cell2dofLocation[0:-1].reshape(-1, 1) + np.arange(smldof)  # (NC,smldof)
        Co[0, idx] = r1  # (NC,NC*Cldof)

        # --- reconstruction matrix
        invls = inv(ls)  # (NC,psmldof,psmldof)
        Csplit = np.hsplit(Co, cell2dofLocation[1:-1])  # list, len(Csplit) is NC, Csplit[i].shape is (psmldof,Cldof)

        f = lambda x: x[0] @ x[1]
        Re = np.concatenate(list(map(f, zip(invls, Csplit))), axis=1)

        return Re

    def stabilizer_matrix(self):
        mesh = self.mesh

    def basis(self, point, index=None, p=None):
        return self.smspace.basis(point, index=index, p=p)

    def grad_basis(self, point, index=None, p=None):
        return self.smspace.grad_basis(point, index=index, p=p)

    def edge_basis(self, point, index=None, p=None):
        return self.smspace.edge_basis(point, index=index, p=p)

    def function(self, dim=None, array=None):
        f = Function(self, dim=dim, array=array)
        return f

    def array(self, dim=None):
        gdof = self.number_of_global_dofs()
        if dim in [None, 1]:
            shape = gdof
        elif type(dim) is int:
            shape = (gdof, dim)
        elif type(dim) is tuple:
            shape = (gdof,) + dim
        return np.zeros(shape, dtype=np.float)
