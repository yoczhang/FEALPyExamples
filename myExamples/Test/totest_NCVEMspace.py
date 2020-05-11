#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: totest_NCVEMspace.py
# @Author: Yongchao Zhang
# @E-mail: yoczhang@126.com
# @Time: May 11, 2020
# ---


from fealpy.functionspace.NonConformingVirtualElementSpace2d import NonConformingVirtualElementSpace2d
import numpy as np
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

ncvem = NonConformingVirtualElementSpace2d(pmesh, p)
ncdof = ncvem.dof
cell2dof, cell2dofLocation = ncdof.cell_to_dof()


# ------------------------------------------------- #
print("End of this test file")
