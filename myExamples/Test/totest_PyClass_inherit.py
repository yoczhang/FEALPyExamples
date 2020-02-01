#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: test_PyClass_inherit.py
# @Author: Yongchao Zhang
# @E-mail: yoczhang@126.com
# @Time: Jan 31, 2020
# ---


# This file is to test the DGSpace2d.py which is inherited from the ScaledMonomialSpace2d.py

import sys
sys.path.append("/Users/yczhang/Documents/FEALPy/FEALPyExamples/myExamples/DG_Poisson")

from DGSpace2d import DGDof2d, DiscontinuousGalerkinSpace2d

import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh.TriangleMesh import TriangleMesh
from fealpy.mesh.Quadtree import Quadtree
from fealpy.mesh.StructureQuadMesh import StructureQuadMesh
from fealpy.mesh.QuadrangleMesh import QuadrangleMesh
from fealpy.functionspace.femdof import CPLFEMDof2d
from fealpy.mesh.mesh_tools import find_node, find_entity
from fealpy.quadrature.GaussLegendreQuadrature import GaussLegendreQuadrature
from fealpy.functionspace.ScaledMonomialSpace2d import SMDof2d, ScaledMonomialSpace2d


# init settings
n = 1  # refine times
p = 2  # polynomial order of FEM space
# q = p + 1  # integration order
q = 2 + 1  # integration order

node = np.array([
    (0, 0),
    (1, 0),
    (1, 1),
    (0, 1)], dtype=np.float)

# ---- tri mesh ----
cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int)  # tri mesh
mesh = TriangleMesh(node, cell)
mesh.uniform_refine(n)
# ------------------

# ---- quad mesh ----
# cell = np.array([(0, 1, 2, 3)], dtype=np.int)  # quad mesh
# # mesh = Quadtree(node, cell)
# mesh = QuadrangleMesh(node, cell)
# mesh.uniform_refine(n)
# -------------------

# -----------
dgdof = DGDof2d(mesh, p)
nlocalcells = dgdof.number_of_local_dofs()


# -----------
dgspace = DiscontinuousGalerkinSpace2d(mesh, p)
massM = dgspace.mass_matrix()

# ------------------------------------------------- #
print("End of this test file")

