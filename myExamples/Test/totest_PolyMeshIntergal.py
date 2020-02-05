#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: totest_PolyMeshIntergal.py
# @Author: Yongchao Zhang
# @E-mail: yoczhang@126.com
# @Time: Feb 04, 2020
# ---

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
from fealpy.mesh.simple_mesh_generator import triangle
from fealpy.quadrature.PolygonMeshIntegralAlg import PolygonMeshIntegralAlg

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
mesh = Quadtree(node, cell)
mesh.uniform_refine(n)
mesh = mesh.to_pmesh()  # Excuse me?! It has this operator!
# -----------------------

# ---- poly mesh ----
# h = 0.2
# box = [0, 1, 0, 1]  # [0, 1]^2 domain
# mesh = triangle(box, h, meshtype='polygon')
# -------------------

# ---- plot mesh ----
fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes, cellcolor='w')
find_entity(axes, mesh, entity='cell', index='all', showindex=True, color='b', markersize=10, fontsize=8)
find_entity(axes, mesh, entity='edge', index='all', showindex=True, color='r', markersize=10, fontsize=8)
find_entity(axes, mesh, entity='node', index='all', showindex=True, color='y', markersize=10, fontsize=8)
plt.show()
# -------------------

# --------
polyInte = PolygonMeshIntegralAlg(mesh, q)


# ----------
bc = mesh.entity_barycenter('cell')
edge2cell = mesh.ds.edge_to_cell()
node = mesh.node
edge = mesh.entity('edge')
tri = [bc[edge2cell[:, 0]], node[edge[:, 0]], node[edge[:, 1]]]

qf = mesh.integrator(q)
bcs, ws = qf.quadpts, qf.weights



pp = np.einsum('ij, jkm->ikm', bcs, tri)

# ------------------------------------------------- #
print("End of this test file")