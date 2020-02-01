#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: totest_edges_to_cells.py
# @Author: Yongchao Zhang
# @E-mail: yoczhang@126.com
# @Time: Feb 01, 2020
# ---


import numpy as np
from fealpy.mesh.TriangleMesh import TriangleMesh
from fealpy.mesh.QuadrangleMesh import QuadrangleMesh
from fealpy.functionspace.ScaledMonomialSpace2d import SMDof2d, ScaledMonomialSpace2d
from fealpy.mesh.simple_mesh_generator import triangle
from fealpy.mesh.mesh_tools import find_entity

import matplotlib.pyplot as plt

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

# # ---- quad mesh ----
# cell = np.array([(0, 1, 2, 3)], dtype=np.int)  # quad mesh
# # mesh = Quadtree(node, cell)
# mesh = QuadrangleMesh(node, cell)
# mesh.uniform_refine(n)
# # -------------------


fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes, cellcolor='w')
find_entity(axes, mesh, entity='cell', index='all', showindex=True, color='b', markersize=16, fontsize=9)
find_entity(axes, mesh, entity='edge', index='all', showindex=True, color='r', markersize=16, fontsize=9)
# find_entity(axes, mesh, entity='node', index='all', showindex=True, color='r', markersize=16, fontsize=9)
# find_node(axes, mesh.node, showindex=True, fontsize=12, markersize=25)
plt.show()


# -------------------
nm = mesh.edge_normal()  # The length of the normal-vector isn't 1, is the length of corresponding edge.

nm_len = np.sqrt(nm[:, 0]**2 + nm[:, 1]**2)

edge_len = mesh.edge_length()

diff_nm_len = nm_len - edge_len


# -------------------
edge2cell = mesh.ds.edge_to_cell()

# ------------------------------------------------- #
print("End of this test file")