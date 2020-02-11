#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: totest_WGSpace.py
# @Author: Yongchao Zhang
# @E-mail: yoczhang@126.com
# @Time: Feb 11, 2020
# ---


import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh.TriangleMesh import TriangleMesh
from fealpy.mesh.Quadtree import Quadtree
from fealpy.mesh.StructureQuadMesh import StructureQuadMesh
from fealpy.mesh.QuadrangleMesh import QuadrangleMesh
from fealpy.functionspace.WeakGalerkinSpace2d import WGDof2d, WeakGalerkinSpace2d
from fealpy.mesh.mesh_tools import find_entity

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
# cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int)  # tri mesh
# mesh = TriangleMesh(node, cell)
# mesh.uniform_refine(n)
# ------------------

# --- quad-tree mesh ---
cell = np.array([(0, 1, 2, 3)], dtype=np.int)  # quad mesh
qtree = Quadtree(node, cell)
qtree.uniform_refine(n)
pmesh = qtree.to_pmesh()  # Excuse me?! It has this operator!
# -----------------------


# ---- plot poly mesh ----
# fig2 = plt.figure()
# axes = fig2.gca()
# pmesh.add_plot(axes, cellcolor='w')
# find_entity(axes, pmesh, entity='cell', index='all', showindex=True, color='b', markersize=10, fontsize=8)
# find_entity(axes, pmesh, entity='edge', index='all', showindex=True, color='r', markersize=10, fontsize=8)
# find_entity(axes, pmesh, entity='node', index='all', showindex=True, color='y', markersize=10, fontsize=8)
# plt.show()
# plt.close()
# -------------------


# --- WG space
wgdof = WGDof2d(pmesh, p)
cell2dof, cell2dofLocation = wgdof.cell_to_dof()
multiIndex1d = wgdof.multi_index_matrix1d()

wgsp = WeakGalerkinSpace2d(pmesh, p)

edge2cell = pmesh.ds.edge2cell
idx = cell2dofLocation[edge2cell[:, [0]]] + edge2cell[:, [2]]*(p+1) + np.arange(p+1)


# ------------------------------------------------- #
print("End of this test file")