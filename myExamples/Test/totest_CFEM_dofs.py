#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: totest_CFEM_dofs.py
# @Author: Yongchao Zhang
# @E-mail: yoczhang@126.com
# @Time: Feb 01, 2020
# ---


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

# init settings
n = 1  # refine times
p = 1  # polynomial order of FEM space
q = p + 1  # integration order
# q = 2 + 1  # integration order

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
node = mesh.entity('node')
edge = mesh.entity('edge')

# fig = plt.figure()
# axes = fig.gca()
# mesh.add_plot(axes, cellcolor='w')
# find_entity(axes, mesh, entity='cell', index='all', showindex=True, color='b', fontsize=15)
# find_entity(axes, mesh, entity='edge', index='all', showindex=True, color='b', fontsize=12)
# plt.show()
# ------------------------------------------------- #


# ------------------------------------------------- #
# ---                 dofs                      --- #
cell2dof = dof.cell2dof
ldof = dof.number_of_local_dofs()
I = np.einsum('k, ij->ijk', np.ones(ldof), cell2dof)
J = I.swapaxes(-1, -2)

I2 = np.einsum('ij, k->ijk',  cell2dof, np.ones(ldof))
J2 = I2.swapaxes(-1, -2)
J2_2 = np.einsum('ij, k->ikj',  cell2dof, np.ones(ldof))


# ------------------------------------------------- #
# ---             integrator                    --- #
integrator = mesh.integrator(q)
qf = integrator
bcs, ws = qf.quadpts, qf.weights
bcs2, ws2 = integrator.get_quadrature_points_and_weights()

# ------------------------------------------------- #
print("End of this test file")

