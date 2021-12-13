#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: FEM_CH_NS_VarCoeff_Model2d.py
# @Author: Yongchao Zhang, Northwest University, Xi'an
# @E-mail: yoczhang@nwu.edu.cn
# @Site:
# @Time: Dec 13, 2021
# ---

__doc__ = """
The FEM for Variable-Coefficient coupled Cahn-Hilliard-Navier-Stokes model in 2D. 
"""

import numpy as np
from scipy.sparse import csr_matrix, spdiags, eye, bmat
from fealpy.quadrature import FEMeshIntegralAlg
from scipy.sparse.linalg import spsolve
from fealpy.boundarycondition import DirichletBC
from fealpy.decorator import timer
from fealpy.functionspace import LagrangeFiniteElementSpace
from sym_diff_basis import compute_basis
from FEM_CH_NS_Model2d import FEM_CH_NS_Model2d


class FEM_CH_NS_VarCoeff_Model2d(FEM_CH_NS_Model2d):
    def __init__(self, pde, mesh, p, dt):
        super(FEM_CH_NS_VarCoeff_Model2d, self).__init__(pde, mesh, p, dt)



    def set_CH_Neumann_edge(self, idxNeuEdge=None):
        """
        We overload the `set_CH_Neumann_edge` function, so that (in the parent class) to call this function.
        :param idxNeuEdge:
        :return:
        """
        if idxNeuEdge is not None:
            return idxNeuEdge

        mesh = self.mesh
        edge2cell = mesh.ds.edge_to_cell()
        bdEdge = (edge2cell[:, 0] == edge2cell[:, 1])  # the bool vars, to get the boundary edges

        isNeuEdge = bdEdge  # here, we first set all the boundary edges are Neu edges
        idxNeuEdge, = np.nonzero(isNeuEdge)  # (NE_Dir,)
        return idxNeuEdge

    def set_NS_Dirichlet_edge(self, idxDirEdge=None):
        """
        We overload the `set_NS_Dirichlet_edge` function, so that (in the parent class) to call this function.
        :param idxDirEdge:
        :return:
        """
        if idxDirEdge is not None:
            return idxDirEdge

        mesh = self.mesh
        edge2cell = mesh.ds.edge_to_cell()
        isBdEdge = (edge2cell[:, 0] == edge2cell[:, 1])  # (NE,), the bool vars, to get the boundary edges

        isDirEdge = isBdEdge  # here, we set all the boundary edges are Dir edges
        idxDirEdge, = np.nonzero(isDirEdge)  # (NE_Dir,)

        return idxDirEdge
