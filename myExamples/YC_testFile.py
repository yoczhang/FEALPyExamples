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

dof = CPLFEMDof2d(mesh, p)

# plot mesh
ipoint = dof.interpolation_points()
cell2dof = dof.cell2dof
node = mesh.entity('node')
edge = mesh.entity('edge')

# fig = plt.figure()
# axes = fig.gca()
# mesh.add_plot(axes, cellcolor='w')
# find_entity(axes, mesh, entity='cell', index='all', showindex=True, color='b', fontsize=15)
# # find_node(axes, ipoint, showindex=False, fontsize=12, markersize=25)
# find_node(axes, node, showindex=True, fontsize=12, markersize=25)
# find_entity(axes, mesh, entity='edge', index='all', showindex=True, color='b', fontsize=12)
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
# ---      project 3:                           --- #
# --- to test ScaledMonomialSpace2d --- matrix_H()
qf = GaussLegendreQuadrature(p + 1)
bcs, ws = qf.quadpts, qf.weights

node = mesh.entity('node')
edge = mesh.entity('edge')
edge2cell = mesh.ds.edge_to_cell()
isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])

node_edge = node[edge]

ps = np.einsum('ij, kjm->ikm', bcs, node[edge])

# --------------
ldof = dof.number_of_local_dofs()
shape = ps.shape[:-1]+(ldof,)
phi0 = np.ones(shape, dtype=np.float)  # (..., M, ldof)
tphi0 = phi0
tphi0[..., 1:3] = phi0[..., 1:3]
start = 3
i = 2
tphi0[..., start:start+i] = phi0[..., start-i:start]*phi0[..., [1]]

# --------------
shape2 = ps.shape[:-1]+(ldof, 2)
gphi = np.ones(shape2, dtype=np.float)
gphi[..., 1, 0] = 1
gphi[..., 2, 1] = 1
len_index = ps.shape[-2]
h = 2*np.ones(len_index, dtype=np.float)
th = h.reshape(-1, 1, 1)
aa = gphi/th

# ------------------------------------------------- #
print("End of this test file")
