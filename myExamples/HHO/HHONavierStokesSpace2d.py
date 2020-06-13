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

        # --- set the row and col index
        CC_row, CC_col, CC_rowedge0, CC_coledge0, CC_rowedge1, CC_coledge1 = self.row_col_trialCell_testCell()
        FC_row0, FC_col0, FC_row1, FC_col1 = self.row_col_trialFace_testCell()
        CF_row0, CF_col0, CF_row1, CF_col1 = self.row_col_trialCell_testFace()

        # ---------------------------------------
        # get the matrix c(u^{k},u^{k+1},v)
        # ---------------------------------------
        def f1(point, index=None):
            phi = self.basis(point, index=index)  # using the cell-integration, so phi: (NQ,NC,vcldof)
            gphi = self.grad_basis(point, index=index)  # using the cell-integration, so gphi: (NQ,NC,vcldof,2)
            valueuh1 = self.value(lastuh1, point, index=index)  # (NQ,NC)
            valueuh2 = self.value(lastuh2, point, index=index)  # (NQ,NC)
            v1 = np.einsum('ijk, ijm, ij->ijmk', gphi[..., 0], phi, valueuh1) \
                + np.einsum('ijk, ijm, ij->ijmk', gphi[..., 1], phi, valueuh2) \
                - np.einsum('ijk, ijm, ij->ijmk', phi, gphi[..., 0], valueuh1) \
                - np.einsum('ijk, ijm, ij->ijmk', phi, gphi[..., 1], valueuh2)  # (NQ,NC,vcldof,vcldof)
            return 0.5 * v1
        block1_trialCell_testCell = self.integralalg.integral(f1, celltype=True)  # (NC,vcldof,vcldof)
        block1_trialFace_testCell0 = 0.5 * np.einsum('i, ijk, ijm, ij->jmk', ws, ephi, phi0,
                                                     uh1celldof_edgevalue0 * n[:, 0]
                                                     + uh2celldof_edgevalue0 * n[:, 1])  # (NE,vcldof,veldof)
        block1_trialFace_testCell1 = 0.5 * np.einsum('i, ijk, ijm, ij->jmk', ws, ephi[:, isInEdge, :], phi1,
                                                     uh1celldof_edgevalue1 * (-n[isInEdge, 0])
                                                     + uh2celldof_edgevalue1 * (-n[isInEdge, 1]))  # (NInE,vcldof,veldof)
        block1_trialCell_testFace0 = -0.5 * np.einsum('i, ijk, ijm, ij->jmk', ws, phi0, ephi,
                                                      uh1celldof_edgevalue0 * n[:, 0]
                                                      + uh2celldof_edgevalue0 * n[:, 1])  # (NE,veldof,vcldof)
        block1_trialCell_testFace1 = -0.5 * np.einsum('i, ijk, ijm, ij->jmk', ws, phi1, ephi[:, isInEdge, :],
                                                      uh1celldof_edgevalue1 * (-n[isInEdge, 0])
                                                      + uh2celldof_edgevalue1 * (-n[isInEdge, 1]))  # (NInE,veldof,vcldof)

        # # get the trialCell_testCell block
        matrix1_trialCell_testCell = csr_matrix((block1_trialCell_testCell.flat,
                                                 (CC_row.flat, CC_col.flat)), shape=(NC*vcldof, NC*vcldof))

        # # get the trialFace_testCell block
        matrix1_trialFace_testCell = csr_matrix((block1_trialFace_testCell0.flat,
                                                 (FC_row0.flat, FC_col0.flat)), shape=(NC*vcldof, NE*veldof))
        matrix1_trialFace_testCell += csr_matrix((block1_trialFace_testCell1.flat,
                                                  (FC_row1.flat, FC_col1.flat)), shape=(NC*vcldof, NE*veldof))

        # # get the trialCell_testFace block
        matrix1_trialCell_testFace = csr_matrix((block1_trialCell_testFace0.flat,
                                                 (CF_row0.flat, CF_col0.flat)), shape=(NE*veldof, NC*vcldof))
        matrix1_trialCell_testFace += csr_matrix((block1_trialCell_testFace1.flat,
                                                  (CF_row1.flat, CF_col1.flat)), shape=(NE*veldof, NC*vcldof))

        # # construct the matrix1
        matrix1_block = bmat([[matrix1_trialCell_testCell, matrix1_trialFace_testCell],
                              [matrix1_trialCell_testFace, None]], format='csr')
        matrix1 = bmat([[matrix1_block, None], [None, matrix1_block]], format='csr')

        # ---------------------------------------
        # get the matrix c(u^{k+1},u^{k},v)
        # ---------------------------------------
        lastUH = [lastuh1, lastuh2]

        def get_matrix2_block(trial_indicator, test_indicator):
            ui = trial_indicator
            vi = test_indicator
            last_uh = lastUH[vi]

            # ----------------------------------
            # get block trialCell_testCell
            # ----------------------------------
            def f2(point, index=None):
                phi = self.basis(point, index=index)  # using the cell-integration, so phi: (NQ,NC,vcldof)
                gphi = self.grad_basis(point, index=index)  # using the cell-integration, so gphi: (NQ,NC,vcldof,2)
                valueuh = self.value(last_uh, point, index=index)  # (NQ,NC)
                gradvalueuh = self.grad_value(last_uh, point, index=index)  # (NQ,NC,2)
                v = np.einsum('ijk, ijm, ij->ijmk', phi, phi, gradvalueuh[..., ui]) \
                    - np.einsum('ijk, ijm, ij->ijmk', phi, gphi[..., ui], valueuh)  # (NQ,NC,vcldof,vcldof)
                return 0.5 * v

            # --- part I --- #
            block2_trialCell_testCell = self.integralalg.integral(f2, celltype=True)  # (NC,vcldof,vcldof)
            matrix2_trialCell_testCell = csr_matrix((block2_trialCell_testCell.flat,
                                                     (CC_row.flat, CC_col.flat)), shape=(NC*vcldof, NC*vcldof))
            # --- part II --- #
            uhedgedof_edgevalue = self.edge_value(last_uh, ps)  # (NQ,NE), using the edge-dofs to get edge-values
            block2_trialCell_testFace0 = 0.5 * np.einsum('i, ijk, ijm, ij, j->jmk', ws, phi0, phi0,
                                                         uhedgedof_edgevalue, n[:, ui])  # (NE,vcldof,vcldof)
            block2_trialCell_testFace1 = 0.5 * np.einsum('i, ijk, ijm, ij, j->jmk', ws, phi1, phi1,
                                                         uhedgedof_edgevalue[:, isInEdge],
                                                         -n[isInEdge, ui])  # (NE,vcldof,vcldof)

            matrix2_trialCell_testCell += csr_matrix((block2_trialCell_testFace0.flat,
                                                      (CC_rowedge0.flat, CC_coledge0.flat)),
                                                     shape=(NC * vcldof, NC * vcldof))
            matrix2_trialCell_testCell += csr_matrix((block2_trialCell_testFace1.flat,
                                                      (CC_rowedge1.flat, CC_coledge1.flat)),
                                                     shape=(NC * vcldof, NC * vcldof))

            # ----------------------------------
            # get block trialCell_testFace
            # ----------------------------------
            uhcelldof0 = lastuh[vcelldof[edge2cell[:, 0], :]]  # (NE,vcldof)
            uhcelldof1 = lastuh[vcelldof[edge2cell[isInEdge, 1], :]]  # (NInE,vcldof)
            uhcelldof_edgevalue0 = np.einsum('ijk, jk->ij', phi0, uhcelldof0)  # (NQ,NE)
            uhcelldof_edgevalue1 = np.einsum('ijk, jk->ij', phi1, uhcelldof1)  # (NQ,NInE)

            block2_trialCell_testFace0 = -0.5 * np.einsum('i, ijk, ijm, ij, j->jmk', ws, phi0, ephi,
                                                          uhcelldof_edgevalue0, n[:, ui])  # (NE,veldof,vcldof)
            block2_trialCell_testFace1 = -0.5 * np.einsum('i, ijk, ijm, ij, j->jmk', ws, phi1, ephi[:, isInEdge, :],
                                                          uhcelldof_edgevalue1, -n[isInEdge, ui])  # (NInE,veldof,vcldof)

            matrix2_trialCell_testFace = csr_matrix((block2_trialCell_testFace0.flat,
                                                     (CF_row0.flat, CF_col0.flat)), shape=(NE*veldof, NC*vcldof))
            matrix2_trialCell_testFace += csr_matrix((block2_trialCell_testFace1.flat,
                                                      (CF_row1.flat, CF_col1.flat)), shape=(NE*veldof, NC*vcldof))

            # ----------------------------------
            # return results
            # ----------------------------------
            return matrix2_trialCell_testCell, matrix2_trialCell_testFace

        # --- to get the different matrix2 --- #
        matrix2_trialCell0_testCell0, matrix2_trialCell0_testFace0 = get_matrix2_block(0, 0)
        matrix2_trialCell1_testCell0, matrix2_trialCell1_testFace0 = get_matrix2_block(1, 0)
        matrix2_trialCell0_testCell1, matrix2_trialCell0_testFace1 = get_matrix2_block(0, 1)
        matrix2_trialCell1_testCell1, matrix2_trialCell1_testFace1 = get_matrix2_block(1, 1)

        block_zero_cell = csr_matrix(np.zeros((NC*vcldof, NE*veldof)))
        block_zero_edge = csr_matrix(np.zeros((NE*veldof, NE*veldof)))

        matrix2 = bmat([[matrix2_trialCell0_testCell0, block_zero_cell, matrix2_trialCell1_testCell0, block_zero_cell],
                        [matrix2_trialCell0_testFace0, block_zero_edge, matrix2_trialCell1_testFace0, block_zero_edge],
                        [matrix2_trialCell0_testCell1, block_zero_cell, matrix2_trialCell1_testCell1, block_zero_cell],
                        [matrix2_trialCell0_testFace1, block_zero_edge, matrix2_trialCell1_testFace1, block_zero_edge]],
                       format='csr')

        # ---------------------------------------
        # get the right vector c(u^{k},u^{k},v)
        # ---------------------------------------
        def get_rightvector_block(test_indicator):
            vi = test_indicator
            last_uh = lastUH[vi]

            # ----------------------------------
            # get block testCell
            # ----------------------------------
            def f3(point, index=None):
                phi = self.basis(point, index=index)  # using the cell-integration, so phi: (NQ,NC,vcldof)
                gphi = self.grad_basis(point, index=index)  # using the cell-integration, so gphi: (NQ,NC,vcldof,2)
                valueuh = self.value(last_uh, point, index=index)  # (NQ,NC)
                valueuh1 = self.value(lastuh1, point, index=index)  # (NQ,NC)
                valueuh2 = self.value(lastuh2, point, index=index)  # (NQ,NC)
                gradvalueuh = self.grad_value(last_uh, point, index=index)  # (NQ,NC,2)
                v = np.einsum('ij, ij, ij->ijmk', gradvalueuh[:, 0], valueuh1, phi) \
                    - np.einsum('ijk, ijm, ij->ijmk', phi, gphi[..., ui], valueuh)  # (NQ,NC,vcldof,vcldof)
                return 0.5 * v

















    def row_col_trialCell_testCell(self):
        p = self.p
        mesh = self.mesh
        NC = mesh.number_of_cells()
        NE = mesh.number_of_edges()
        edge2cell = mesh.ds.edge_to_cell()
        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
        InEdgeIdx, = np.nonzero(isInEdge)
        NInE = len(InEdgeIdx)

        vcldof = self.dof.vDof.number_of_cell_local_dof()
        veldof = p + 1  # number of edge local dof

        # --- construct the row and col of cell-dofs at cells --- #
        row_atCell = np.einsum('i, j->ij', range(NC * vcldof),
                        np.ones(vcldof)).reshape((NC, vcldof, vcldof))  # (NC,vcldof,vcldof)
        col_atCell = row_atCell.swapaxes(1, -1)

        # --- construct the row and col of cell-dofs at faces --- #
        r0 = ((vcldof * edge2cell[:, 0]).reshape(-1, 1) + np.tile(np.arange(vcldof), (NE, 1))).flatten()
        row_atEdge0 = np.einsum('i, j->ij', r0, np.ones(vcldof)).reshape((NE, vcldof, vcldof))  # (NE,vcldof,vcldof)
        col_atEdge0 = row_atEdge0.swapaxes(1, -1)

        r1 = ((vcldof * edge2cell[isInEdge, 0]).reshape(-1, 1) + np.tile(np.arange(vcldof), (NInE, 1))).flatten()
        row_atEdge1 = np.einsum('i, j->ij', r1, np.ones(vcldof)).reshape((NInE, vcldof, vcldof))  # (NInE,vcldof,vcldof)
        col_atEdge1 = row_atEdge1.swapaxes(1, -1)

        return row_atCell, col_atCell, row_atEdge0, col_atEdge0, row_atEdge1, col_atEdge1

    def row_col_trialFace_testCell(self):
        p = self.p
        mesh = self.mesh
        NC = mesh.number_of_cells()
        NE = mesh.number_of_edges()
        edge2cell = mesh.ds.edge_to_cell()
        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
        InEdgeIdx, = np.nonzero(isInEdge)
        NInE = len(InEdgeIdx)

        vcldof = self.dof.vDof.number_of_cell_local_dof()
        veldof = p + 1  # number of edge local dof

        r0 = ((vcldof * edge2cell[:, 0]).reshape(-1, 1) + np.tile(np.arange(vcldof), (NE, 1))).flatten()
        row0 = np.einsum('i, j->ij', r0, np.ones(veldof)).reshape((NE, vcldof, veldof))  # (NE,vcldof,veldof)
        c0 = np.tile((np.arange(NE * veldof)).reshape(1, -1), (vcldof, 1))
        c0 = np.hsplit(c0, np.arange(veldof, NE * veldof, veldof))
        col0 = np.array(c0)  # (NE,vcldof,veldof)

        r1 = ((vcldof * edge2cell[isInEdge, 1]).reshape(-1, 1) + np.tile(np.arange(vcldof), (NInE, 1))).flatten()
        row1 = np.einsum('i, j->ij', r1, np.ones(veldof)).reshape((NInE, vcldof, veldof))
        c1 = np.tile(((veldof * InEdgeIdx).reshape(-1, 1) + np.tile(np.arange(veldof), (NInE, 1))).flatten(),
                     (vcldof, 1))
        c1 = np.hsplit(c1, np.arange(veldof, NInE * veldof, veldof))
        col1 = np.array(c1)  # (NInE,vcldof,veldof)
        return row0, col0, row1, col1

    def row_col_trialCell_testFace(self):
        p = self.p
        mesh = self.mesh
        NC = mesh.number_of_cells()
        NE = mesh.number_of_edges()
        edge2cell = mesh.ds.edge_to_cell()
        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
        InEdgeIdx, = np.nonzero(isInEdge)
        NInE = len(InEdgeIdx)

        vcldof = self.dof.vDof.number_of_cell_local_dof()
        veldof = p + 1  # number of edge local dof

        r0 = ((veldof * np.arange(NE)).reshape(-1, 1) + np.tile(np.arange(veldof), (NE, 1))).flatten()
        row0 = np.einsum('i, j->ij', r0, np.ones(vcldof)).reshape((NE, veldof, vcldof))  # (NE,veldof,vclodf)
        c0 = np.tile(((vcldof * edge2cell[:, 0]).reshape(-1, 1) +
                      np.tile(np.arange(vcldof), (NE, 1))).flatten(), (veldof, 1))
        c0 = np.hsplit(c0, np.arange(vcldof, NE * vcldof, vcldof))
        col0 = np.array(c0)

        r1 = ((veldof * InEdgeIdx).reshape(-1, 1) + np.tile(np.arange(veldof), (NInE, 1))).flatten()
        row1 = np.einsum('i, j->ij', r1, np.ones(vcldof)).reshape((NInE, veldof, vcldof))  # (NE,veldof,vclodf)
        c1 = np.tile(((vcldof * edge2cell[isInEdge, 1]).reshape(-1, 1) +
                      np.tile(np.arange(vcldof), (NInE, 1))).flatten(), (veldof, 1))
        c1 = np.hsplit(c1, np.arange(vcldof, NInE * vcldof, vcldof))
        col1 = np.array(c1)
        return row0, col0, row1, col1

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












