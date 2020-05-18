#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: totest_VectorSpace.py
# @Author: Yongchao Zhang
# @E-mail: yoczhang@126.com
# @Time: May 18, 2020
# ---

from fealpy.functionspace.LagrangeFiniteElementSpace import LagrangeFiniteElementSpace
from fealpy.mesh.mesh_tools import find_node, find_entity
from fealpy.mesh.simple_mesh_generator import triangle
from scipy import sparse
import matplotlib.pyplot as plt
import numpy as np


# init settings
n = 1  # refine times
p = 1  # polynomial order of FEM space
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

# --- poly mesh ---
h = 0.6
box = [0, 1, 0, 1]  # [0, 1]^2 domain
pmesh = triangle(box, h, meshtype='polygon')
# -----------------

# --- quad-tree mesh ---
# cell = np.array([(0, 1, 2, 3)], dtype=np.int)  # quad mesh
# qtree = Quadtree(node, cell)
# qtree.uniform_refine(n)
# pmesh = qtree.to_pmesh()  # Excuse me?! It has this operator!
# -----------------------

# ---- plot poly mesh ----
# fig2 = plt.figure()
# axes = fig2.gca()
# pmesh.add_plot(axes, cellcolor='w')
# find_entity(axes, pmesh, entity='cell', index=None, showindex=True, color='b', markersize=10, fontsize=8)
# find_entity(axes, pmesh, entity='edge', index=None, showindex=True, color='r', markersize=10, fontsize=8)
# find_entity(axes, pmesh, entity='node', index=None, showindex=True, color='y', markersize=10, fontsize=8)
# plt.show()
# plt.close()
# -------------------

mesh = pmesh

scalarspace = LagrangeFiniteElementSpace(mesh, p, spacetype='C')
# TD = scalarspace.TD
# GD = scalarspace.GD
TD = 2
GD = 2

# phi: (NQ,NC,ldof)
NQ = 6
NC = 4
ldof = 3

phi = np.arange(NQ*NC*ldof).reshape((NQ, NC, ldof))

shape = list(phi.shape[:-1])
phi1 = np.einsum('...j, mn->...jmn', phi, np.eye(GD))
shape += [-1, GD]
phi2 = phi1.reshape(shape)

aa = np.arange(9).reshape(3, 3)
aa[1, 1] = 0
aa[2, 2] = 0
aa.astype(object)
gdof = 3
S = sparse.coo_matrix(aa)
I, J = np.nonzero(S)
A = sparse.coo_matrix((2*gdof, 2*gdof), dtype=mesh.ftype)
for i in range(GD):
    A += sparse.coo_matrix((S.data, (GD * I + i, GD * J + i)), shape=(2*gdof, 2*gdof), dtype=mesh.ftype)

bb = A.toarray()

# ------------------------------------------------- #
print("End of this test file")







