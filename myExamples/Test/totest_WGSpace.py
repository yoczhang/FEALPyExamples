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
from fealpy.functionspace.ScaledMonomialSpace2d import SMDof2d, ScaledMonomialSpace2d
from fealpy.quadrature import GaussLegendreQuadrature
from fealpy.quadrature import PolygonMeshIntegralAlg

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


# #
# #
# --- WG space test --- #

smspace = ScaledMonomialSpace2d(pmesh, p)
wgdof = WGDof2d(pmesh, p)
wgsp = WeakGalerkinSpace2d(pmesh, p)
integralalg = smspace.integralalg

multiIndex1d = wgdof.multi_index_matrix1d()

# smldof = smspace.number_of_local_dofs()
# cell2dof, cell2dofLocation = wgdof.cell2dof, wgdof.cell2dofLocation
# R0 = np.zeros((smldof, len(cell2dof)), dtype=np.float)
# R1 = np.zeros((smldof, len(cell2dof)), dtype=np.float)
#
# edge2cell = pmesh.ds.edge2cell
# idx = cell2dofLocation[edge2cell[:, [0]]] + edge2cell[:, [2]] * (p + 1) + np.arange(p + 1)

# --- test begin --- #
node = pmesh.entity('node')
edge = pmesh.entity('edge')
edge2cell = pmesh.ds.edge_to_cell()
isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])

h = integralalg.edgemeasure
n = pmesh.edge_unit_normal()

qf = GaussLegendreQuadrature(p + 3)
bcs, ws = qf.quadpts, qf.weights
ps = np.einsum('ij, kjm->ikm', bcs, node[edge])
phi0 = smspace.basis(ps, index=edge2cell[:, 0])
phi1 = smspace.basis(ps[:, isInEdge, :], index=edge2cell[isInEdge, 1])
phi = smspace.edge_basis(ps)

F0 = np.einsum('i, ijm, ijn, j->mjn', ws, phi0, phi, h)
F1 = np.einsum('i, ijm, ijn, j->mjn', ws, phi1, phi[:, isInEdge, :], h[isInEdge])

smldof = smspace.number_of_local_dofs()
cell2dof, cell2dofLocation = wgdof.cell2dof, wgdof.cell2dofLocation
R0 = np.zeros((smldof, len(cell2dof)), dtype=np.float)
R1 = np.zeros((smldof, len(cell2dof)), dtype=np.float)

idx = cell2dofLocation[edge2cell[:, [0]]] + edge2cell[:, [2]] * (p + 1) + np.arange(p + 1)
R0[:, idx] = n[np.newaxis, :, [0]] * F0
R1[:, idx] = n[np.newaxis, :, [1]] * F0
if isInEdge.sum() > 0:
    idx1 = cell2dofLocation[edge2cell[isInEdge, 1]].reshape(-1, 1) \
          + (p + 1) * edge2cell[isInEdge, [3]].reshape(-1, 1) + np.arange(p + 1)
    n = n[isInEdge]
    R0[:, idx1] = -n[np.newaxis, :, [0]] * F1  # 这里应该加上负号
    R1[:, idx1] = -n[np.newaxis, :, [1]] * F1  # 这里应该加上负号


def f(x, index):
    gphi = smspace.grad_basis(x, index)
    phi = smspace.basis(x, index)
    return np.einsum('...mn, ...k->...nmk', gphi, phi)


M = integralalg.integral(f, celltype=True)
idof = (p + 1) * (p + 2) // 2
idx = cell2dofLocation[1:].reshape(-1, 1) + np.arange(-idof, 0)
R0[:, idx] = -M[:, 0].swapaxes(0, 1)
R1[:, idx] = -M[:, 1].swapaxes(0, 1)

# ------------------------------------------------- #
print("End of this test file")
