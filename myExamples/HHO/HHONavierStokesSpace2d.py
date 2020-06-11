#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: HHONavierStokesSpace2d.py
# @Author: Yongchao Zhang
# @E-mail: yoczhang@126.com
# @Time: Jun 04, 2020
# ---


import numpy as np
from numpy.linalg import inv
# from fealpy.common import block, block_diag
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye, bmat

# from fealpy.functionspace.function import Function
from fealpy.quadrature import GaussLegendreQuadrature
# from fealpy.quadrature import PolygonMeshIntegralAlg
# from fealpy.functionspace.ScaledMonomialSpace2d import SMDof2d, ScaledMonomialSpace2d
from myScaledMonomialSpace2d import SMDof2d, ScaledMonomialSpace2d
from HHOStokesSpace2d import HHOStokesDof2d, HHOStokesSpace2d


class HHONavierStokesDof2d:
    def __init__(self, mesh, p):
        self.mesh = mesh
        self.p = p
        self.vDof = self.velocityDof()
        self.pDof = self.pressureDof()

    def velocityDof(self):
        # # note that, this Dof only has the scalar Dof
        return HHOStokesDof2d(self.mesh, self.p).velocityDof()

    def pressureDof(self):
        return HHOStokesDof2d(self.mesh, self.p).pressureDof()

    def number_of_global_dofs(self):
        return 2 * self.vDof.number_of_global_dofs() + self.pDof.number_of_global_dofs()


class HHONavierStokesSpace2d:
    def __init__(self, mesh, p):
        self.mesh = mesh
        self.p = p
        self.dof = HHONavierStokesDof2d(mesh, p)
        self.stokesspace = HHOStokesSpace2d(mesh, p)
        self.vSpace = self.stokesspace.vSpace
        self.pSpace = self.stokesspace.pSpace
        self.integralalg = self.vSpace.integralalg

    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def system_matrix(self, nu):
        A = self.stokesspace.velocity_matrix()  # (2*vgdof,2*vgdof)
        B = self.stokesspace.divergence_matrix()  # (pgdof,2*vgdof)
        P = self.stokesspace.pressure_correction()  # (1,2*vgdof+pgdof)

    def convective_matrix(self, lastuh):
        """
        To solve the Navier-Stokes equation, we need the Newton iteration,
        this function is designed to get the matrices uesed in the Newton iteration.
        :param lastuh: last step solution uh,
        :return:
        """
        p = self.p
        mesh = self.mesh
        NC = mesh.number_of_cells()
        NE = mesh.number_of_edges()
        node = mesh.entity('node')
        edge = mesh.entity('edge')
        edge2cell = mesh.ds.edge_to_cell()
        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
        InEdgeIdx, = np.nonzero(isInEdge)
        NInE = len(InEdgeIdx)

        # hE = self.integralalg.edgemeasure  # (NE,), the length of edges
        n = mesh.edge_normal()  # (NE,2), the normal vector of edges, and its norm is the length of corresponding edges
        # # The direction of normal vector is from edge2cell[i,0] to edge2cell[i,1]
        # # (that is, from the cell with smaller number to the cell with larger number).

        qf = GaussLegendreQuadrature(p + 3)
        bcs, ws = qf.quadpts, qf.weights  # bcs.shape: (NQ,2); ws.shape: (NQ,)
        ps = np.einsum('ij, kjm->ikm', bcs, node[edge])  # (NQ,NE,2), NE is the number of edges

        phi0 = self.basis(ps, index=edge2cell[:, 0], p=p)  # (NQ,NE,vcldof)
        phi1 = self.basis(ps[:, isInEdge, :], index=edge2cell[isInEdge, 1], p=p)  # (NQ,NInE,vcldof)
        ephi = self.edge_basis(ps)  # (NQ,NE,veldof), veldof is the number of local 1D dofs on one edge

        # --- the last uh settings --- #
        vDof = self.dof.vDof
        vgdof = vDof.number_of_global_dofs()
        vcldof = vDof.number_of_cell_local_dof()
        veldof = p + 1  # number of edge local dof
        vcelldof = np.arange(NC*vcldof).reshape(NC, vcldof)
        lastuh1 = np.squeeze(lastuh[:vgdof])  # (vgdof,)
        lastuh2 = np.squeeze(lastuh[vgdof:])  # (vgdof,)

        # uh1edgedof_edgevalue = self.edge_value(lastuh1, ps)  # (NQ,NE), using the edge-dofs to get edge-values
        # uh2edgedof_edgevalue = self.edge_value(lastuh2, ps)  # (NQ,NE), using the edge-dofs to get edge-values
        uh1celldof0 = lastuh1[vcelldof[edge2cell[:, 0], :]]  # (NE,vcldof)
        uh1celldof1 = lastuh1[vcelldof[edge2cell[isInEdge, 1], :]]  # (NInE,vcldof)
        uh2celldof0 = lastuh2[vcelldof[edge2cell[:, 0], :]]  # (NE,vcldof)
        uh2celldof1 = lastuh2[vcelldof[edge2cell[isInEdge, 1], :]]  # (NInE,vcldof)
        uh1celldof_edgevalue0 = np.einsum('ijk, jk->ij', phi0, uh1celldof0)  # (NQ,NE)
        uh1celldof_edgevalue1 = np.einsum('ijk, jk->ij', phi1, uh1celldof1)  # (NQ,NInE)
        uh2celldof_edgevalue0 = np.einsum('ijk, jk->ij', phi0, uh2celldof0)  # (NQ,NE)
        uh2celldof_edgevalue1 = np.einsum('ijk, jk->ij', phi1, uh2celldof1)  # (NQ,NInE)

        # cell2dof, cell2dofLocation = self.dof.vDof.cell_to_dof()
        # uh1split = np.split(lastuh1[cell2dof], cell2dofLocation[1:-1])  # list, each-term shape: (Cldof,)
        # uh2split = np.split(lastuh2[cell2dof], cell2dofLocation[1:-1])  # list, each-term shape: (Cldof,)

        # --- get the matrix c(u^{k},u^{k+1},v) --- #
        def f1(point, index=None):
            phi = self.basis(point, index=index)  # using the cell-integration, so phi: (NQ,NC,vcldof)
            gphi = self.grad_basis(point, index=index)  # using the cell-integration, so gphi: (NQ,NC,vcldof,2)
            valueuh1 = self.value(lastuh1, point, index=index)  # (NQ,NC)
            valueuh2 = self.value(lastuh2, point, index=index)  # (NQ,NC)
            r1 = np.einsum('ijk, ijm, ij->ijmk', gphi[..., 0], phi, valueuh1) \
                + np.einsum('ijk, ijm, ij->ijmk', gphi[..., 1], phi, valueuh2) \
                - np.einsum('ijk, ijm, ij->ijmk', phi, gphi[..., 0], valueuh1) \
                - np.einsum('ijk, ijm, ij->ijmk', phi, gphi[..., 1], valueuh2)  # (NQ,NC,vcldof,vcldof)
            return r1/2.0
        matrix1_trialCell_testCell = self.integralalg.integral(f1, celltype=True)  # (NC,vcldof,vcldof)
        matrix1_trialFace_testCell0 = 0.5 * np.einsum('i, ijk, ijm, ia, ib, a, b->jmk', ws, ephi, phi0,
                                                      uh1celldof_edgevalue0, uh2celldof_edgevalue0,
                                                      n[:, 0], n[:, 1])  # (NE,vcldof,veldof)
        matrix1_trialFace_testCell1 = 0.5 * np.einsum('i, ijk, ijm, ia, ib, a, b->jmk', ws, ephi[:, isInEdge, :], phi1,
                                                      uh1celldof_edgevalue1, uh2celldof_edgevalue1,
                                                      n[isInEdge, :][:, 0], n[isInEdge, :][:, 1])  # (NInE,vcldof,veldof)
        matrix1_trialCell_testFace0 = -0.5 * np.einsum('i, ijk, ijm, ia, ib, a, b->jmk', ws, phi0, ephi,
                                                       uh1celldof_edgevalue0, uh2celldof_edgevalue0,
                                                       n[:, 0], n[:, 1])  # (NE,veldof,vcldof)
        matrix1_trialCell_testFace1 = -0.5 * np.einsum('i, ijk, ijm, ia, ib, a, b->jmk', ws, phi1, ephi[:, isInEdge, :],
                                                       uh1celldof_edgevalue1, uh2celldof_edgevalue1,
                                                       n[isInEdge, :][:, 0], n[isInEdge, :][:, 1])  # (NInE,veldof,vcldof)

        # # get the trialCell_testCell block
        block1_row = np.einsum('i, j->ij', range(NC*vcldof), np.ones(vcldof)).reshape((NC, vcldof, vcldof))
        block1_col = block1_row.swapaxes(1, -1)
        block1_trialCell_testCell = csr_matrix((matrix1_trialCell_testCell.flat,
                                                (block1_row.flat, block1_col.flat)), shape=(NC*vcldof, NC*vcldof))

        # # get the trialFace_testCell block
        r0 = ((vcldof*edge2cell[:, 0]).reshape(-1, 1) + np.tile(np.arange(vcldof), (NE, 1))).flatten()
        block1_row0 = np.einsum('i, j->ij', r0, np.ones(veldof)).reshape((NE, vcldof, veldof))  # (NE,vcldof,veldof)
        c0 = np.tile((np.arange(NE*veldof)).reshape(1, -1), (vcldof, 1))
        c0 = np.hsplit(c0, np.arange(veldof, NE*veldof, veldof))
        block1_col0 = np.array(c0)  # (NE,vcldof,veldof)

        r1 = ((vcldof * edge2cell[isInEdge, 1]).reshape(-1, 1) + np.tile(np.arange(vcldof), (NInE, 1))).flatten()
        block1_row1 = np.einsum('i, j->ij', r1, np.ones(veldof)).reshape((NInE, vcldof, veldof))
        c1 = np.tile((np.arange(NInE * veldof)).reshape(1, -1), (vcldof, 1))
        c1 = np.hsplit(c1, np.arange(veldof, NInE * veldof, veldof))
        block1_col1 = np.array(c1)  # (NInE,vcldof,veldof)

        block1_trialFace_testCell = csr_matrix((matrix1_trialFace_testCell0.flat,
                                                (block1_row0.flat, block1_col0.flat)), shape=(NC*vcldof, NE*veldof))
        block1_trialFace_testCell += csr_matrix((matrix1_trialFace_testCell1.flat,
                                                 (block1_row1.flat, block1_col1.flat)), shape=(NC*vcldof, NE*veldof))

        # # get the trialCell_testFace block




















        # def get_convective_matrix(x):
        #     celluh1 = x[0]
        #     celluh2 = x[1]
        #     cellIdx = x[2]  # the cell index of current cell
        #     cellNE = x[3]  # the number of edges in current cell
        #
        #     def f1(point, index=None):
        #         h = 0





    def basis(self, point, index=None, p=None):
        return self.vSpace.basis(point, index=index, p=p)

    def grad_basis(self, point, index=None, p=None):
        return self.vSpace.grad_basis(point, index=index, p=p)

    def value(self, uh, point, index=None):
        NC = self.mesh.number_of_cells()
        smldof = self.vSpace.smldof
        return self.vSpace.value(uh[:NC*smldof, ...], point, index=index)

    def grad_value(self, uh, point, index=None):
        NC = self.mesh.number_of_cells()
        smldof = self.vSpace.smldof
        return self.vSpace.grad_value(uh[:NC*smldof, ...], point, index=index)

    def edge_basis(self, point, index=None, p=None):
        return self.vSpace.edge_basis(point, index=index, p=p)

    def edge_value(self, uh, point):
        # point: (NQ,NE,2), NQ is the number of quadrature points, NE is the number of edges
        ephi = self.edge_basis(point)  # (NQ,NE,eldof), eldof is the number of local 1D dofs on one edge
        edge2dof = self.dof.vDof.edge_to_dof()

        dim = len(uh.shape) - 1
        s0 = 'abcdefg'[:dim]
        s1 = '...ij, ij{}->...i{}'.format(s0, s0)
        val = np.einsum(s1, ephi, uh[edge2dof])  # (NQ,NE) or (NQ,NE,1)
        return val












