#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: totest_CapillaryWaveMesh_fromMat.py
# @Author: Yongchao Zhang
# @Institution: Northwest University, Xi'an, Shaanxi, China
# @E-mail: yoczhang@126.com, yoczhang@nwu.edu.cn
# @Site: 
# @Time: Feb 15, 2022
# ---

import numpy as np
from scipy.io import loadmat
from fealpy.mesh.TriangleMesh import TriangleMesh
from fealpy.mesh.HalfEdgeMesh2d import HalfEdgeMesh2d
import matplotlib  # 为了解决画图时采用 GUI (plt.show()) 的形式时, python3.8 崩溃的情况.
matplotlib.use("TkAgg")  # 为了解决画图时采用 GUI (plt.show()) 的形式时, python3.8 崩溃的情况.
import matplotlib.pyplot as plt
import os, sys
os.chdir(sys.path[0])

print(os.getcwd())
mfile = loadmat('./CapillaryWaveMesh_2.mat')
node = mfile['node']
cell = mfile['elem']
mesh = TriangleMesh(node, cell)
mesh = HalfEdgeMesh2d.from_mesh(mesh, NV=3)

Nrefine = 2
# for k in range(Nrefine):
#     bc = mesh.cell_barycenter()
#     NC = mesh.number_of_cells()
#     cellstart = mesh.ds.cellstart
#     isMarkedCell = np.zeros(NC + cellstart, dtype=np.bool_)
#     isMarkedCell[cellstart:] = abs(bc[:, 1] - 0.) < 0.015
#     # mesh.refine_triangle_rg(isMarkedCell)
#     mesh.refine_triangle_nvb(isMarkedCell)
#     bc1 = mesh.cell_barycenter()
#     NC1 = mesh.number_of_cells()
#     cc, = np.nonzero(abs(bc[:, 1] - 0.) < 0.015)

node = mesh.node
cell = mesh.ds.cell_to_node()
np.save('./WaveMeshNode_mat3', node)
np.save('./WaveMeshCell_mat3', cell)

# fig = plt.figure()
# axes = fig.gca()
# mesh.add_plot(axes)
# # mesh.find_cell(axes, showindex=True)
# plt.show()
# plt.close()

print('end of the file')

