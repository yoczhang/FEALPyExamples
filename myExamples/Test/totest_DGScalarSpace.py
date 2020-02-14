#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: test_PyClass_inherit.py
# @Author: Yongchao Zhang
# @E-mail: yoczhang@126.com
# @Time: Jan 31, 2020
# ---


# This file is to test the DGScalarSpace2d.py which is inherited from the ScaledMonomialSpace2d.py


from DGScalarSpace2d import DGScalarDof2d, DGScalarSpace2d
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

import sys
sys.path.append("/Users/yczhang/Documents/FEALPy/FEALPyExamples/myExamples/DG_Poisson")



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

# ---- poly mesh ----
h = 0.2
box = [0, 1, 0, 1]  # [0, 1]^2 domain
mesh = triangle(box, h, meshtype='polygon')
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


# -----------
dgdof = DGScalarDof2d(mesh, p)
cell2dof = dgdof.cell2dof
ldof = dgdof.number_of_local_dofs()
edge = mesh.entity('edge')

# -----------
dgspace = DGScalarSpace2d(mesh, p)
massM = dgspace.mass_matrix()
AJ, JA, JJ = dgspace.interiorEdge_matrix()


# -----------
qf = GaussLegendreQuadrature(p + 1)  # the integral points on edges (1D)
bcs, ws = qf.quadpts, qf.weights  # bcs.shape: (NQ,2); ws.shape: (NQ,)
NQ = len(ws)

# -----------


# -----------
NE = mesh.number_of_edges()
N = NE*NQ*ldof
phi0 = np.ones(N).reshape(NQ, NE, ldof)
phi1 = 3*np.ones(N).reshape(NQ, NE, ldof)

test_edge_area = 0.1*np.arange(1, NE+1)
phyws = np.einsum('i,j->ij', ws, test_edge_area)

Jmm = np.einsum('ij, ijk, ijm->jmk', phyws, phi0, phi1)  # Jmm.shape: (NInE,ldof,ldof)

Jmm_1 = np.einsum('i, ijk, ijm, j->jmk', ws, phi0, phi1, test_edge_area)  # Jmm.shape: (NInE,ldof,ldof)

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

