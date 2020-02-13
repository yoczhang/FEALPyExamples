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
from numpy.linalg import inv
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
phi0 = smspace.basis(ps, index=edge2cell[:, 0])  # (NQ,NE,smldof)
phi1 = smspace.basis(ps[:, isInEdge, :], index=edge2cell[isInEdge, 1])  # (NQ,NInE,smldof)
phi = smspace.edge_basis(ps)  # (NQ,NE,ldof1d)

F0 = np.einsum('i, ijm, ijn, j->mjn', ws, phi0, phi, h)  # (smldof,NE,ldof1d)
F1 = np.einsum('i, ijm, ijn, j->mjn', ws, phi1, phi[:, isInEdge, :], h[isInEdge])

smldof = smspace.number_of_local_dofs()
cell2dof, cell2dofLocation = wgdof.cell2dof, wgdof.cell2dofLocation
R0 = np.zeros((smldof, len(cell2dof)), dtype=np.float)
# # R0.shape: (smldof,NC*CNldof), CNldof is the number of all dofs in one cell
R1 = np.zeros((smldof, len(cell2dof)), dtype=np.float)

idx = cell2dofLocation[edge2cell[:, [0]]] + edge2cell[:, [2]] * (p + 1) + np.arange(p + 1)  # (NE,ldof1d)
R0[:, idx] = n[np.newaxis, :, [0]] * F0
R1[:, idx] = n[np.newaxis, :, [1]] * F0
if isInEdge.sum() > 0:
    idx1 = cell2dofLocation[edge2cell[isInEdge, 1]].reshape(-1, 1) \
           + (p + 1) * edge2cell[isInEdge, [3]].reshape(-1, 1) + np.arange(p + 1)
    n = n[isInEdge]
    R0[:, idx1] = -n[np.newaxis, :, [0]] * F1  # 这里应该加上负号
    R1[:, idx1] = -n[np.newaxis, :, [1]] * F1  # 这里应该加上负号


def f(x, index):
    gphi = smspace.grad_basis(x, index)  # (NQ,NInE,smldof,2)
    phi = smspace.basis(x, index)  # (NQ,NInE,smldof)
    # the return: (2,smldof,smldof)
    return np.einsum('...mn, ...k->...nmk', gphi, phi)


M = integralalg.integral(f, celltype=True)  # (NC,2,smldof,smldof)
idof = (p + 1) * (p + 2) // 2
idx = cell2dofLocation[1:].reshape(-1, 1) + np.arange(-idof, 0)  # (NC,smldof)
R0[:, idx] = -M[:, 0].swapaxes(0, 1)  # M[:, 0].shape (NC,smldof,smldof)
R1[:, idx] = -M[:, 1].swapaxes(0, 1)

tR0 = np.hsplit(R0, cell2dofLocation[1:-1])

# --- weak gradient --- #
CM = smspace.cell_mass_matrix()
EM = smspace.edge_mass_matrix()

H0 = inv(CM)
H1 = inv(EM)

cd = np.hsplit(cell2dof, cell2dofLocation[1:-1])
r0 = np.hsplit(R0, cell2dofLocation[1:-1])
r1 = np.hsplit(R1, cell2dofLocation[1:-1])
ph = smspace.function(dim=2)

uh = np.ones(len(cell2dof))
f0 = lambda x: x[0] @ (x[1] @ uh[x[2]])
ph[:, 0] = np.concatenate(list(map(f0, zip(H0, r0, cd))))
# # Here,
# # H0 is array: (NC,smldof,smldof);
# # r0 is list: len(r0)=4, r0[0].shape=(smldof,CNdof), CNldof is the number of all dofs in one cell
# # cd is list: len(cd)=4, cd[0].shape=(CNdof,)
# # Every-time the x of f0 is x=(H0[i,...], r0[i], cd[i]), i = 0,1,2,3,
# # i.e., x[0] is H0[i,...]; x[1] is r0[i]; x[2] is cd[i].


ph[:, 1] = np.concatenate(list(map(f0, zip(H0, r1, cd))))

# ------------------------------------------------- #
print("End of this test file")
