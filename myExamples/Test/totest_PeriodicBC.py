#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: totest_PeriodicBC.py
# @Author: Yongchao Zhang
# @Institution: Northwest University, Xi'an, Shaanxi, China
# @E-mail: yoczhang@126.com, yoczhang@nwu.edu.cn
# @Site: 
# @Time: Jan 23, 2022
# ---


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
from fealpy.mesh import MeshFactory as MF
from fealpy.mesh.HalfEdgeMesh2d import HalfEdgeMesh2d

# init settings
n = 2  # refine times
p = 1  # polynomial order of FEM space
# q = p + 1  # integration order
q = 2 + 1  # integration order

# ---- tri mesh ----
# node = np.array([
#     (0, 0),
#     (1, 0),
#     (1, 1),
#     (0, 1)], dtype=np.float)
# cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int)  # tri mesh
# mesh = TriangleMesh(node, cell)
# mesh.uniform_refine(n)
# mesh = HalfEdgeMesh2d.from_mesh(mesh, NV=3)  # 三角形网格的单边数据结构
# ------------------

# |--- other mesh
domain = [0, 1, 0, 1]
nn = 4
mesh = MF.boxmesh2d(domain, nx=nn, ny=nn, meshtype='tri')
# mesh = HalfEdgeMesh2d.from_mesh(mesh, NV=3)  # 三角形网格的单边数据结构

dof = CPLFEMDof2d(mesh, p)


ipoint = dof.interpolation_points()
cell2dof = dof.cell2dof
edge2dof = dof.edge_to_dof()
node = mesh.entity('node')
edge = mesh.entity('edge')

# |--- plot mesh
# fig = plt.figure()
# axes = fig.gca()
# mesh.add_plot(axes, cellcolor='w')
# find_entity(axes, mesh, entity='cell', showindex=True, color='b', fontsize=15)
# # find_node(axes, ipoint, showindex=False, fontsize=12, markersize=25)
# find_node(axes, node, showindex=True, fontsize=12, markersize=25)
# find_entity(axes, mesh, entity='edge', showindex=True, color='b', fontsize=12)
# plt.show()

# |--- setting left and right boundary as the periodic boundary
bc_indx = mesh.ds.boundary_edge_index()
e2n = mesh.ds.edge_to_node()[bc_indx, :]

bc_mid_coor_x = 0.5 * (node[e2n[:, 0], 0] + node[e2n[:, 1], 0])
periBC_idx0 = bc_indx[np.abs(bc_mid_coor_x - 0.) < 1.e-8]
periBC_idx1 = bc_indx[np.abs(bc_mid_coor_x - 1.) < 1.e-8]



print('end of the file')

