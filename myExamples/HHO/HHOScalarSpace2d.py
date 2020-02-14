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


