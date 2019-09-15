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
from fealpy.functionspace.dof import CPLFEMDof2d
from fealpy.mesh.mesh_tools import find_node, find_entity

# init settings
n = 1  # refine times
p = 2  # polynomial order of FEM space
q = p + 1  # integration order

node = np.array([
    (0, 0),
    (1, 0),
    (1, 1),
    (0, 1)], dtype=np.float)

# cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int)  # tri mesh
# mesh = TriangleMesh(node, cell)
# mesh.uniform_refine(n)

cell = np.array([(0, 1, 2, 3)], dtype=np.int)
mesh = Quadtree(node, cell)
mesh.uniform_refine(n)

dof = CPLFEMDof2d(mesh, p)

# plot mesh
ipoint = dof.interpolation_points()
cell2dof = dof.cell2dof

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes, cellcolor='w')
find_entity(axes, mesh, entity='cell', index='all', showindex=True, color='b', fontsize=15)
# find_node(axes, ipoint, showindex=False, fontsize=12, markersize=25)
plt.show()

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
bc = bcs
ftype = mesh.ftype
TD = 2  # topological dimension
multiIndex = dof.multiIndex

c = np.arange(1, p+1, dtype=np.int)
P = 1.0/np.multiply.accumulate(c)
t = np.arange(0, p)
shape = bc.shape[:-1]+(p+1, TD+1)
A = np.ones(shape, dtype=ftype)
A[..., 1:, :] = p*bc[..., np.newaxis, :] - t.reshape(-1, 1)
np.cumprod(A, axis=-2, out=A)
A[..., 1:, :] *= P.reshape(-1, 1)
idx = np.arange(TD+1)
phi = np.prod(A[..., multiIndex, idx], axis=-1)

print(phi.shape)


# ------------------------------------------------- #
print("End of this test file")
