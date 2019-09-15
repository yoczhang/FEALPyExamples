#!/anaconda3/envs/FEALPy/bin python3.7
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site:
# @File: test.py
# @Author: Yongchao Zhang
# @E-mail: yoczhang@126.com
# @Time: Aug 15, 2019
# ---

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, spdiags
from fealpy.functionspace.function import Function
from fealpy.functionspace.lagrange_fem_space import LagrangeFiniteElementSpace
from fealpy.functionspace.dof import CPLFEMDof1d, CPLFEMDof2d, CPLFEMDof3d
from fealpy.functionspace.dof import DPLFEMDof1d, DPLFEMDof2d, DPLFEMDof3d


class DiscontinuousGalerkinSpace(LagrangeFiniteElementSpace):
    def __init__(self, mesh, p=1):
        spacetype = 'D'
        super(DiscontinuousGalerkinSpace, self).__init__(mesh, p, spacetype)

    def __str__(self):
        return "Discontinuous Galerkin finite element space!"

    def left_matrix(self, uh, bc, cellidx=None):
        phi = self.basis(bc)

