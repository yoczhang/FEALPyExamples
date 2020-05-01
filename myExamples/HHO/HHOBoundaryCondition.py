#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: HHOBoundaryCondition.py
# @Author: Yongchao Zhang
# @E-mail: yoczhang@126.com
# @Time: May 01, 2020
# ---

import numpy as np


class HHOBoundaryCondition:
    def __init__(self, space):
        self.space = space
        self.mesh = space.mesh
        self.p = space.p
        self.eldof = self.p + 1
        self.NE = self.mesh.number_of_edges()
        self.egdof = self.NE * self.eldof

    def set_Dirichlet_edge(self):
        mesh = self.mesh
        edge2cell = mesh.ds.edge_to_cell()
        bdEdge = (edge2cell[:, 0] == edge2cell[:, 1])  # the bool vars, to get the boundary edges

        isDirEdge = bdEdge  # here, we set all the boundary edges are Dir edges

        return isDirEdge

    def set_Neumann_edge(self):
        mesh = self.mesh
        edge2cell = mesh.ds.edge_to_cell()
        bdEdge = (edge2cell[:, 0] == edge2cell[:, 1])  # the bool vars, to get the boundary edges

        isNeuEdge = bdEdge  # here, we first set all the boundary edges are Neu edges

        issetNeuEdge = 'no'
        if issetNeuEdge == 'no':
            isNeuEdge = None

        return isNeuEdge

    def set_Dirichlet_dof(self):
        NE = self.NE
        eldof = self.eldof
        egdof = self.egdof

        isDirDof = np.zeros(egdof).astype(np.bool)  # 1-D array, (egdof,)

        isDirEdge = self.set_Dirichlet_edge()
        idxDirEdge = np.arange(NE)[isDirEdge]
        DirDof = eldof * idxDirEdge.reshape(-1, 1) + np.arange(eldof)
        DirDof = np.squeeze(DirDof.reshape(1, -1))  # np.squeeze transform 2-D array (NDirDof,1) into 1-D (NDirDof,)
        isDirDof[DirDof] = True  # 1-D array, (egdof,)

        return isDirDof

    def set_Neumann_dof(self):
        NE = self.NE
        eldof = self.eldof
        egdof = self.egdof

        isNeuDof = np.zeros(egdof).astype(np.bool)  # 1-D array, (egdof,)

        isNeuEdge = self.set_Neumann_edge()

        if isNeuEdge is not None:
            idxNeuEdge = np.arange(NE)[isNeuEdge]
            NeuDof = eldof * idxNeuEdge.reshape(-1, 1) + np.arange(eldof)
            NeuDof = np.squeeze(NeuDof.reshape(1, -1))  # np.squeeze transform 2-D array (NNeuDof,1) into 1-D (NNeuDof,)
            isNeuDof[NeuDof] = True  # 1-D array, (egdof,)

        return isNeuDof

    def set_Free_dof(self):
        isDirDof = self.set_Dirichlet_dof()
        isNeuDof = self.set_Neumann_dof()

        isFreeDof = ~(isDirDof + isNeuDof)

        return isFreeDof


