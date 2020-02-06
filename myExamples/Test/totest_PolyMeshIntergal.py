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
from fealpy.pde.sfc_2d import SFCModelData0
from fealpy.functionspace import ConformingVirtualElementSpace2d


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
mesh = qtree.to_pmesh()  # Excuse me?! It has this operator!
# -----------------------

# ---- poly mesh ----
# h = 0.2
# box = [0, 1, 0, 1]  # [0, 1]^2 domain
# mesh = triangle(box, h, meshtype='polygon')
# -------------------

# ---- plot mesh ----
# fig = plt.figure()
# axes = fig.gca()
# mesh.add_plot(axes, cellcolor='w')
# find_entity(axes, mesh, entity='cell', index='all', showindex=True, color='b', markersize=10, fontsize=8)
# find_entity(axes, mesh, entity='edge', index='all', showindex=True, color='r', markersize=10, fontsize=8)
# find_entity(axes, mesh, entity='node', index='all', showindex=True, color='y', markersize=10, fontsize=8)
# plt.show()
# -------------------

# --------
pde = SFCModelData0()
space = ConformingVirtualElementSpace2d(mesh, p)
smspace = ScaledMonomialSpace2d(mesh, p)
f = pde.source
phi = smspace.basis


def u(x, index):
    return np.einsum('ij, ijm->ijm', f(x), phi(x, index=index))


def triangle_measure(tri):
    v1 = tri[1] - tri[0]
    v2 = tri[2] - tri[0]
    area = np.cross(v1, v2)/2
    return area


polyInte = PolygonMeshIntegralAlg(mesh, q)


# ----------
bc = mesh.entity_barycenter('cell')
edge2cell = mesh.ds.edge_to_cell()
node = mesh.node
edge = mesh.entity('edge')
tri = [bc[edge2cell[:, 0]], node[edge[:, 0]], node[edge[:, 1]]]
# # tri is a list-type var.
# # tri[0].shape: (NE,2). Is the coordinates of 0-th points of the triangles.
# # tri[1]                Is the coordinates of 1-th points of the triangles.
# # tri[2]                Is the coordinates of 2-th points of the triangles.

a = triangle_measure(tri)
NC = mesh.number_of_cells()

qf = mesh.integrator(q)
bcs, ws = qf.quadpts, qf.weights  # bcs.shape: (NQ,tdim+1).   ws.shape: (NQ,)

pp = np.einsum('ij, jkm->ikm', bcs, tri)  # pp.shape: (NQ,NE,2)
# # this is from the formula: the physical point:
# # \vec{x} = \vec{x}_0\lambda_0 + \vec{x}_1\lambda_1 + \vec{x}_2\lambda_2,
# # where \vec{x} = (x, y)^T.

val = u(pp, edge2cell[:, 0])  # val.shape: (NQ,NE,smsldof), smsldof is the number smsspace local dof.

shape = (NC, ) + val.shape[2:]  # shape.shape: (NC,smsldof)
e = np.zeros(shape, dtype=np.float)

ee = np.einsum('i, ij..., j->j...', ws, val, a)  # ee.shape: (NE,smsldof)
np.add.at(e, edge2cell[:, 0], ee)

isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])

tri = [
    bc[edge2cell[isInEdge, 1]],
    node[edge[isInEdge, 1]],
    node[edge[isInEdge, 0]]
    ]
a = triangle_measure(tri)
pp = np.einsum('ij, jkm->ikm', bcs, tri)
val = u(pp, edge2cell[isInEdge, 1])
ee = np.einsum('i, ij..., j->j...', ws, val, a)
np.add.at(e, edge2cell[isInEdge, 1], ee)

e1 = e.sum(axis=0)

# ------------------------------------------------- #
print("End of this test file")
