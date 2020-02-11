#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: totest_refine_qtree.py
# @Author: Yongchao Zhang
# @E-mail: yoczhang@126.com
# @Time: Feb 10, 2020
# ---

import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh.TriangleMesh import TriangleMesh
from fealpy.mesh.Quadtree import Quadtree
from fealpy.mesh.StructureQuadMesh import StructureQuadMesh
from fealpy.mesh.QuadrangleMesh import QuadrangleMesh
from fealpy.functionspace.femdof import CPLFEMDof2d
from fealpy.mesh.mesh_tools import find_node, find_entity

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

# ---- quad mesh ----
# cell = np.array([(0, 1, 2, 3)], dtype=np.int)  # quad mesh
# mesh = QuadrangleMesh(node, cell)
# mesh.uniform_refine(n)
# -------------------

# --- quad-tree mesh ---
cell = np.array([(0, 1, 2, 3)], dtype=np.int)  # quad mesh
qtree = Quadtree(node, cell)
qtree.uniform_refine(n)
pmesh = qtree.to_pmesh()  # Excuse me?! It has this operator!
# -----------------------

# ---- poly mesh ----
# h = 0.2
# box = [0, 1, 0, 1]  # [0, 1]^2 domain
# mesh = triangle(box, h, meshtype='polygon')
# -------------------

# ---- plot qtree mesh ----
# fig1 = plt.figure()
# axes = fig1.gca()
# qtree.add_plot(axes, cellcolor='w')
# find_entity(axes, qtree, entity='cell', index='all', showindex=True, color='b', markersize=10, fontsize=8)
# find_entity(axes, qtree, entity='edge', index='all', showindex=True, color='r', markersize=10, fontsize=8)
# find_entity(axes, qtree, entity='node', index='all', showindex=True, color='y', markersize=10, fontsize=8)
# plt.show()
# plt.close()
# -------------------

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

# --- refine --- #
for i in range(1):
    leafCellIdx = qtree.leaf_cell_index()
    Nleaf = len(leafCellIdx)
    isMarked = np.zeros(Nleaf, dtype=np.bool)
    isMarked[1] = True
    isMarked[-1] = True

    data = {}
    uh = np.ones(pmesh.number_of_nodes())
    data[0] = uh

    NC_qtree = qtree.number_of_cells()
    markedCellInd = leafCellIdx[isMarked]
    isMarkedCell = np.zeros(NC_qtree, dtype=np.bool)
    isMarkedCell[markedCellInd] = True
    qtree.refine(isMarkedCell, data=data)
    # parent_refinedqtree = qtree.parent
    cellLocation = pmesh.ds.cellLocation

    # --- coarsen --- #
    leafCellIdx = qtree.leaf_cell_index()
    Nleaf = len(leafCellIdx)
    isMarked = np.zeros(Nleaf, dtype=np.bool)
    isMarked[-1] = True

    NC_qtree = qtree.number_of_cells()
    markedCellInd = leafCellIdx[isMarked]
    isMarkedCell = np.zeros(NC_qtree, dtype=np.bool)
    isMarkedCell[markedCellInd] = True
    qtree.coarsen(isMarkedCell)

    # ---- plot qtree mesh ----
    # fig1 = plt.figure()
    # axes = fig1.gca()
    # qtree.add_plot(axes, cellcolor='w')
    # find_entity(axes, qtree, entity='cell', index='all', showindex=True, color='b', markersize=10, fontsize=8)
    # find_entity(axes, qtree, entity='edge', index='all', showindex=True, color='r', markersize=10, fontsize=8)
    # find_entity(axes, qtree, entity='node', index='all', showindex=True, color='y', markersize=10, fontsize=8)
    # plt.show()
    # plt.close()
    # -------------------

pmesh = qtree.to_pmesh()

# ---- plot poly mesh ----
fig2 = plt.figure()
axes = fig2.gca()
pmesh.add_plot(axes, cellcolor='w')
find_entity(axes, pmesh, entity='cell', index='all', showindex=True, color='b', markersize=10, fontsize=8)
find_entity(axes, pmesh, entity='edge', index='all', showindex=True, color='r', markersize=10, fontsize=8)
find_entity(axes, pmesh, entity='node', index='all', showindex=True, color='y', markersize=10, fontsize=8)
plt.show()
plt.close()
# -------------------


# ------------------------------------------------- #
print("End of this test file")
