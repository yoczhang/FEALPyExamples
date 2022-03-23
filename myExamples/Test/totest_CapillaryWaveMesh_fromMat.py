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


def find_lastSecondMin(A):
    As = np.sort(A)
    B, = np.nonzero(abs(As - As[0]) < 1.e-8)
    theLastSecond = len(B)
    return As[theLastSecond]


def find_special_refined_cell(mesh, measure, y_measure):
    cb = mesh.cell_barycenter()
    cb_x = cb[:, 0]
    cb_y = cb[:, 1]

    outflag = abs(cb_y - 0.) < y_measure
    aa = np.arange(0, 1, measure) - measure/2.
    current_cell = []
    for k in range(len(aa)-1):
        aa_m = aa[k+1]
        flag = (abs(cb_x - aa_m) < 1.e-8) & outflag
        current_cell_temp, = np.nonzero(flag)
        current_cell.append(current_cell_temp)
    the_cell = np.array(current_cell)
    return the_cell.flatten()


print(os.getcwd())
mfile = loadmat('./CapillaryWaveMesh_4.mat')
node = mfile['node']
cell = mfile['elem']
mesh = TriangleMesh(node, cell)
mesh = HalfEdgeMesh2d.from_mesh(mesh, NV=3)

# cell = np.array([[0, 1, 3], [4, 3, 1], [4, 1, 5], [2, 5, 1],
#                  [6, 3, 7], [4, 7, 3], [4, 5, 7], [8, 7, 5],
#                  [6, 7, 9], [10, 9, 7], [10, 7, 11], [8, 11, 7],
#                  [12, 9, 13], [10, 13, 9], [10, 11, 13], [14, 13, 11]], dtype=np.int)
# node = np.array([[0, -1], [0.5, -1], [1, -1],
#                  [0, -0.5], [0.5, -0.5], [1, -0.5],
#                  [0, 0], [0.5, 0], [1, 0],
#                  [0, 0.5], [0.5, 0.5], [1, 0.5],
#                  [0, 1], [0.5, 1], [1, 1]], dtype=np.float)
# mesh = TriangleMesh(node, cell)
# mesh = HalfEdgeMesh2d.from_mesh(mesh, NV=3)
# # fig = plt.figure()
# # axes = fig.gca()
# # mesh.add_plot(axes)
# # # mesh.find_cell(axes, showindex=True)
# # plt.show()
# # plt.close()

toRefineDomain = [0.03, 0.03]
Nrefine = len(toRefineDomain)
for k in range(Nrefine):
    bc = mesh.cell_barycenter()
    NC = mesh.number_of_cells()
    cellstart = mesh.ds.cellstart
    isMarkedCell = np.zeros(NC + cellstart, dtype=np.bool_)
    cellmeasure = mesh.entity_measure('cell')
    if k == 0:
        current_cell = find_special_refined_cell(mesh, 1./24, 0.03)
        isMarkedCell[current_cell + 1] = True
        mesh.refine_triangle_nvb(isMarkedCell)
        node = np.array(mesh.node)
        cell = mesh.ds.cell_to_node()
        mesh = TriangleMesh(node, cell)
        mesh = HalfEdgeMesh2d.from_mesh(mesh, NV=3)
    else:
        isMarkedCell[cellstart:] = abs(bc[:, 1] - 0.) <= toRefineDomain[k]
        mesh.refine_triangle_rg(isMarkedCell)

    print('mesh.number_of_cells() = ', mesh.number_of_cells())
    # bc1 = mesh.cell_barycenter()
    # NC1 = mesh.number_of_cells()
    # cc, = np.nonzero(abs(bc[:, 1] - 0.) < 0.015)

node = mesh.node
cell = mesh.ds.cell_to_node()
np.save('./WaveMeshNode_mat5', node)
np.save('./WaveMeshCell_mat5', cell)

# fig = plt.figure()
# axes = fig.gca()
# mesh.add_plot(axes)
# # mesh.find_cell(axes, showindex=True)
# plt.show()
# plt.close()

print('end of the file')

