#!/usr/bin/env python3
#
# yc test file
#

# ------------------------------------------------- #
# --- project 1: get barycentric points (bcs)   --- #

import sys
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
from fealpy.mesh.HalfEdgeMesh2d import HalfEdgeMesh2d

# |--- init settings
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
mesh = HalfEdgeMesh2d.from_mesh(mesh, NV=3)  # 三角形网格的单边数据结构
# ------------------

# ---- quad mesh ----
# cell = np.array([(0, 1, 2, 3)], dtype=np.int)  # quad mesh
# # mesh = Quadtree(node, cell)
# mesh = QuadrangleMesh(node, cell)
# mesh.uniform_refine(n)
# -------------------

dof = CPLFEMDof2d(mesh, p)

# plot mesh
ipoint = dof.interpolation_points()
cell2dof = dof.cell2dof
edge2dof = dof.edge_to_dof()
node = mesh.entity('node')
edge = mesh.entity('edge')

# fig = plt.figure()
# axes = fig.gca()
# mesh.add_plot(axes, cellcolor='w')
# find_entity(axes, mesh, entity='cell', showindex=True, color='b', fontsize=15)
# # find_node(axes, ipoint, showindex=False, fontsize=12, markersize=25)
# find_node(axes, node, showindex=True, fontsize=12, markersize=25)
# find_entity(axes, mesh, entity='edge', showindex=True, color='b', fontsize=12)
# plt.show()

# get bcs
integrator = mesh.integrator(q)
qf = integrator
bcs, ws = qf.quadpts, qf.weights
shape = bcs.shape

print(shape)
# ------------------------------------------------- #

# ------------------------------------------------- #
# ---      project 2: get basis at bcs          --- #

# Ref: lagrange_fem_space.py -- basis
bcs = bcs
ftype = mesh.ftype
TD = 2  # topological dimension
multiIndex = dof.multiIndex

c = np.arange(1, p + 1, dtype=np.int)
P = 1.0 / np.multiply.accumulate(c)
t = np.arange(0, p)
shape = bcs.shape[:-1] + (p + 1, TD + 1)
A = np.ones(shape, dtype=ftype)
A[..., 1:, :] = p * bcs[..., np.newaxis, :] - t.reshape(-1, 1)
np.cumprod(A, axis=-2, out=A)
A[..., 1:, :] *= P.reshape(-1, 1)
idx = np.arange(TD + 1)
phi = np.prod(A[..., multiIndex, idx], axis=-1)

print(phi.shape)


# ------------------------------------------------- #
print("End of this test file")
