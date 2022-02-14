#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: totest_meshrefine.py
# @Author: Yongchao Zhang
# @Institution: Northwest University, Xi'an, Shaanxi, China
# @E-mail: yoczhang@126.com, yoczhang@nwu.edu.cn
# @Site: 
# @Time: Jan 29, 2022
# ---

import numpy as np
import sys
import time
# from prst.gridprocessing import *
from ..Tools.mesh_IO import mesh_IO
from fealpy.mesh import MeshFactory as MF
from fealpy.mesh.HalfEdgeMesh2d import HalfEdgeMesh2d
from fealpy.mesh import Quadtree
from fealpy.mesh import TriangleMesh, PolygonMesh, QuadrangleMesh
import matplotlib  # 为了解决画图时采用 GUI (plt.show()) 的形式时, python3.8 崩溃的情况.
matplotlib.use("TkAgg")  # 为了解决画图时采用 GUI (plt.show()) 的形式时, python3.8 崩溃的情况.
import matplotlib.pyplot as plt

# |--- init mesh
# box = [0, 1, 0, 1]
# mesh = MF.boxmesh2d(box, nx=4, ny=4, meshtype='tri')
# mesh = HalfEdgeMesh2d.from_mesh(mesh, NV=3)  # 三角形网格的单边数据结构
# fig = plt.figure()
# axes = fig.gca()
# mesh.add_plot(axes)
# mesh.find_cell(axes, showindex=True)
# plt.show()
# plt.close()

# |--- refine mesh
# NC = mesh.number_of_cells()
# cellstart = mesh.ds.cellstart
# isMarkedCell = np.zeros(NC + cellstart, dtype=np.bool_)
# isMarkedCell[cellstart+0] = True
# isMarkedCell[cellstart+3] = True
# mesh.refine_triangle_rg(isMarkedCell)

# |--- quad mesh
# cell = np.array([[0, 1, 2, 3], [1, 4, 5, 2]], dtype=np.int)
# node = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [2, 0], [2, 1]], dtype=np.float)
# mesh = QuadrangleMesh(node, cell)
# mesh.uniform_refine(2)
# mesh = HalfEdgeMesh2d.from_mesh(mesh, NV=4)

# |--- test prst
# G = cartGrid(np.array([4, 5]))

# |--- special tri mesh 0
# cell = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4],
#                  [3, 2, 7], [2, 5, 7], [5, 6, 7], [6, 3, 7]], dtype=np.int)
# node = np.array([[0, -1], [1, -1], [1, 0], [0, 0], [0.5, -0.5],
#                  [1, 1], [0, 1], [0.5, 0.5]], dtype=np.float)
# mesh = TriangleMesh(node, cell)
# mesh = HalfEdgeMesh2d.from_mesh(mesh, NV=3)
# mesh.uniform_refine(2)
# bc = mesh.cell_barycenter()
# NC = mesh.number_of_cells()
# cellstart = mesh.ds.cellstart
# isMarkedCell = np.zeros(NC + cellstart, dtype=np.bool_)
# isMarkedCell[cellstart+4] = True
# mesh.refine_triangle_rg(isMarkedCell)

# |--- special tri mesh 1
cell = np.array([[0, 1, 3], [4, 3, 1], [4, 1, 5], [2, 5, 1],
                 [6, 3, 7], [4, 7, 3], [4, 5, 7], [8, 7, 5],
                 [6, 7, 9], [10, 9, 7], [10, 7, 11], [8, 11, 7],
                 [12, 9, 13], [10, 13, 9], [10, 11, 13], [14, 13, 11]], dtype=np.int)
node = np.array([[0, -1], [0.5, -1], [1, -1],
                 [0, -0.5], [0.5, -0.5], [1, -0.5],
                 [0, 0], [0.5, 0], [1, 0],
                 [0, 0.5], [0.5, 0.5], [1, 0.5],
                 [0, 1], [0.5, 1], [1, 1]], dtype=np.float)
mesh = TriangleMesh(node, cell)
mesh = HalfEdgeMesh2d.from_mesh(mesh, NV=3)
# bc = mesh.cell_barycenter()
# NC = mesh.number_of_cells()
# cellstart = mesh.ds.cellstart
# isMarkedCell = np.zeros(NC + cellstart, dtype=np.bool_)
# isMarkedCell[cellstart+4] = True
# isMarkedCell[cellstart:] = abs(bc[:, 1] - 0.) < 0.2
# mesh.refine_triangle_rg(isMarkedCell)
mesh.uniform_refine(2)

mesh_IO.save2MatlabMesh(mesh, './CapillaryWaveInitMesh')


# |--- plot the mesh
# fig = plt.figure()
# axes = fig.gca()
# mesh.add_plot(axes)
# # mesh.find_cell(axes, showindex=True)
# plt.show()
# # plt.close()

# |--- according to the physical-barycenter of cells
cm = 1.
k = 0
tt = 0.2
while cm > 0.01/2:
    bc = mesh.cell_barycenter()
    NC = mesh.number_of_cells()
    cellstart = mesh.ds.cellstart
    isMarkedCell = np.zeros(NC + cellstart, dtype=np.bool_)
    isMarkedCell[cellstart:] = abs(bc[:, 1] - 0.) < tt
    mesh.refine_triangle_rg(isMarkedCell)
    # if k < 2:
    #     mesh.uniform_refine(1)

    # bc = mesh.cell_barycenter()
    # NC = mesh.number_of_cells()
    # cellstart = mesh.ds.cellstart
    # isMarkedCell = np.zeros(NC + cellstart, dtype=np.bool_)
    # isMarkedCell[cellstart:] = (abs(bc[:, 1] - 0.) < tt) & (abs(bc[:, 0] - 0.5) < tt/4)
    # mesh.refine_triangle_rg(isMarkedCell)

    fig = plt.figure()
    axes = fig.gca()
    mesh.add_plot(axes)
    # mesh.find_cell(axes, showindex=True)
    plt.show()
    plt.close()

    k += 1
    print('|--- k = %d ---|' % k)
    cm = np.sqrt(np.min(mesh.entity_measure('cell')))
    if tt > 0.025:
        tt = tt / 2.






print('end of the file')
