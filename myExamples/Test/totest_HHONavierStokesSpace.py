#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: totest_HHONavierStokesSpace.py
# @Author: Yongchao Zhang
# @E-mail: yoczhang@126.com
# @Time: Jun 10, 2020
# ---


from HHONavierStokesSpace2d import HHONavierStokesSpace2d
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
# from fealpy.functionspace.ScaledMonomialSpace2d import SMDof2d, ScaledMonomialSpace2d
from myScaledMonomialSpace2d import ScaledMonomialSpace2d

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
# pmesh.ds.cell_to_edge()
# edge2cell = pmesh.ds.edge2cell

# #
# #
# --- HHO space setting --- #
smspace = ScaledMonomialSpace2d(pmesh, p)
integralalg = smspace.integralalg
nsspace = HHONavierStokesSpace2d(pmesh, p)

# --- test begin --- #
lastuh = nsspace.vSpace.function()
lastuh[:] = np.random.rand(len(lastuh))
lastuh = np.concatenate([lastuh, 2.0 + lastuh])
cm = nsspace.convective_matrix(lastuh)

# np.add.at(aa, [[0, 0, 2, 2], [1, 3, 1, 3]], bb.flatten())


# ------------------------------------------------- #
print("End of this test file")


