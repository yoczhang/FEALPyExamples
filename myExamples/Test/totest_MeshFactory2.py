#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: totest_MeshFactory2.py
# @Author: Yongchao Zhang, Northwest University, Xi'an
# @E-mail: yoczhang@nwu.edu.cn
# @Site:
# @Time: May 19, 2022
# ---

from fealpy.mesh import MeshFactory as MF
import numpy as np
import matplotlib  # 为了解决画图时采用 GUI (plt.show()) 的形式时, python3.8 崩溃的情况.
matplotlib.use("TkAgg")  # 为了解决画图时采用 GUI (plt.show()) 的形式时, python3.8 崩溃的情况.
import matplotlib.pyplot as plt


n = 4
p = 1
box = [-1.5, 1.5, 0, 1]  # [0, 1]^2 domain


def boxmesh2d_rice(box, nx=10, ny=10):
    qmesh = MF.boxmesh2d(box, nx=nx, ny=ny, meshtype='quad')
    node = qmesh.entity('node')
    cell = qmesh.entity('cell')
    NN = qmesh.number_of_nodes()
    NE = qmesh.number_of_edges()
    NC = qmesh.number_of_cells()

    isLeftCell = np.zeros((nx, ny), dtype=np.bool)
    isLeftCell[0, 0::2] = True
    isLeftCell[1, 1::2] = True
    if nx > 2:
        isLeftCell[2::2, :] = isLeftCell[0, :]
    if nx > 3:
        isLeftCell[3::2, :] = isLeftCell[1, :]
    isLeftCell = isLeftCell.reshape(-1)

    lcell = cell[isLeftCell]
    rcell = cell[~isLeftCell]
    newCell = np.r_['0', 
                    lcell[:, [1, 2, 0]], 
                    lcell[:, [3, 0, 2]],
                    rcell[:, [0, 1, 3]],
                    rcell[:, [2, 3, 1]]]
    return MF.TriangleMesh(node, newCell)


mesh = boxmesh2d_rice(box, nx=3*n, ny=n)


# |--- plot the mesh
fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
# mesh.find_cell(axes, showindex=True)
plt.show()
plt.close()

