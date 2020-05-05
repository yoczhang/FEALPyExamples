#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: totest_HHOScalarSpace.py
# @Author: Yongchao Zhang
# @E-mail: yoczhang@126.com
# @Time: Feb 14, 2020
# ---


from HHOScalarSpace2d import HHODof2d, HHOScalarSpace2d
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh.TriangleMesh import TriangleMesh
from fealpy.mesh.Quadtree import Quadtree
from fealpy.mesh.StructureQuadMesh import StructureQuadMesh
from fealpy.mesh.QuadrangleMesh import QuadrangleMesh
from fealpy.functionspace.femdof import CPLFEMDof2d
from fealpy.mesh.mesh_tools import find_node, find_entity
from fealpy.quadrature.GaussLegendreQuadrature import GaussLegendreQuadrature
from fealpy.mesh.simple_mesh_generator import triangle
from fealpy.functionspace.ScaledMonomialSpace2d import SMDof2d, ScaledMonomialSpace2d


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
# find_entity(axes, pmesh, entity='cell', index='all', showindex=True, color='b', markersize=10, fontsize=8)
# find_entity(axes, pmesh, entity='edge', index='all', showindex=True, color='r', markersize=10, fontsize=8)
# find_entity(axes, pmesh, entity='node', index='all', showindex=True, color='y', markersize=10, fontsize=8)
# plt.show()
# plt.close()
# -------------------
# pmesh.ds.cell_to_edge()
# edge2cell = pmesh.ds.edge2cell

# #
# #
# --- HHO space setting --- #
smspace = ScaledMonomialSpace2d(pmesh, p)
integralalg = smspace.integralalg
hhospace = HHOScalarSpace2d(pmesh, p)
hhodof = HHODof2d(pmesh, p)
edge_to_dof = hhodof.edge_to_dof()
cell2dof, doflocation = hhodof.cell_to_dof()
multiIndex1d = hhodof.multi_index_matrix1d()


# --- test begin --- #
psm2sm = hhospace.projection_psmspace_to_smspace()
CRM = hhospace.construct_righthand_matrix()
Re = hhospace.reconstruction_matrix()
StiffM = hhospace.reconstruction_stiff_matrix()
Pp2s = hhospace.projection_psmspace_to_smspace()
F, pF = hhospace.projection_sm_psm_space_to_edge()
StabM = hhospace.reconstruction_stabilizer_matrix()

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
ephi = smspace.edge_basis(ps)  # (NQ,NE,ldof1d)

cell2dof, cell2dofLocation = hhospace.dof.cell2dof, hhospace.dof.cell2dofLocation
idx = cell2dofLocation[edge2cell[:, [0]]] + edge2cell[:, [2]] * (p + 1) + np.arange(p + 1)  # (NE,eldof)

# ------------------------------------------------- #
print("End of this test file")