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
        edge2dof = np.arange(NE*(p+1)).reshape(NE, p+1)
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
        cell2dofLocation = np.zeros(NC+1, dtype=np.int)
        cell2dofLocation[1:] = np.add.accumulate(ldof)
        cell2dof = np.zeros(cell2dofLocation[-1], dtype=np.int)

        edge2dof = self.edge_to_dof()
        edge2cell = mesh.ds.edge_to_cell()
        idx = cell2dofLocation[edge2cell[:, [0]]] + edge2cell[:, [2]]*(p+1) + np.arange(p+1)
        cell2dof[idx] = edge2dof

        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
        idx = (cell2dofLocation[edge2cell[isInEdge, 1]] + edge2cell[isInEdge, 3]*(p+1)).reshape(-1, 1) + np.arange(p+1)
        cell2dof[idx] = edge2dof[isInEdge]

        NV = mesh.number_of_vertices_of_cells()
        NE = mesh.number_of_edges()
        idof = (p+1)*(p+2)//2
        idx = (cell2dofLocation[:-1] + NV*(p+1)).reshape(-1, 1) + np.arange(idof)
        cell2dof[idx] = NE*(p+1) + np.arange(NC*idof).reshape(NC, idof)
        return cell2dof, cell2dofLocation

    def number_of_global_dofs(self):
        p = self.p
        mesh = self.mesh
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()
        gdof = NE*(p+1) + NC*(p+1)*(p+2)//2
        return gdof

    def number_of_local_dofs(self):
        p = self.p
        mesh = self.mesh
        NCE = mesh.number_of_edges_of_cells()
        ldofs = NCE*(p+1) + (p+1)*(p+2)//2
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
        if doftype is 'all':
            return self.dof.cell2dof, self.dof.cell2dofLocation
        elif doftype is 'cell':
            p = self.p
            NE = self.mesh.number_of_edges()
            NC = self.mesh.number_of_cells()
            idof = (p+1)*(p+2)//2
            cell2dof = NE*(p+1) + np.arange(NC*idof).reshape(NC, idof)
            return cell2dof

    def reconstruction_matrix(self):
        p = self.p
        mesh = self.mesh
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
        # # phi0, phi1 are the potential variable, are trial functions, taking order p;
        # # pphi0, pphi1 are the test functions, taking order p+1.
        phi0 = self.basis(ps, index=edge2cell[:, 0])  # (NQ,NE,smldof)
        phi1 = self.basis(ps[:, isInEdge, :], index=edge2cell[isInEdge, 1])

        gpphi0 = self.grad_basis(ps, index=edge2cell[:, 0], p=p+1)  # (NQ,NE,smldof,2)
        gpphi1 = self.grad_basis(ps[:, isInEdge, :], index=edge2cell[isInEdge, 1], p=p+1)  # (NQ,NInE,smldof,2)



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
            shape = (gdof, ) + dim
        return np.zeros(shape, dtype=np.float)





