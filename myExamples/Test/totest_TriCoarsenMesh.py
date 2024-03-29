#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: totest_TriCoarsenMesh.py
# @Author: Yongchao Zhang, Northwest University, Xi'an
# @E-mail: yoczhang@nwu.edu.cn
# @Site:
# @Time: Jun 10, 2022
# ---


from fealpy.mesh import MeshFactory as MF
import numpy as np
import matplotlib  # 为了解决画图时采用 GUI (plt.show()) 的形式时, python3.8 崩溃的情况.
matplotlib.use("TkAgg")  # 为了解决画图时采用 GUI (plt.show()) 的形式时, python3.8 崩溃的情况.
import matplotlib.pyplot as plt


def boxmesh2d_rice(box, nx=10, ny=10):
    qmesh = MF.boxmesh2d(box, nx=nx, ny=ny, meshtype='quad')
    node = qmesh.entity('node')
    cell = qmesh.entity('cell')

    isLeftCell = np.zeros((nx, ny), dtype=bool)
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


# |--- to test ---| #
if __name__ == '__main__':
    box = [-1.5, 1.5, 0, 1]
    mesh = boxmesh2d_rice(box, nx=4 * 3, ny=4)
    ax = mesh.add_plot(plt)
    mesh.find_cell(ax.axes, showindex=True)
    # plt.show()
    plt.savefig('mesh-0.png')
    plt.close()

    # |--- coarsen
    isMarkedCell = np.zeros(mesh.number_of_cells(), dtype=bool)
    # mk = [17, 42, 88, 67]
    mk = [17, 67]
    isMarkedCell[mk] = True
    mesh.coarsen(isMarkedCell=isMarkedCell)
    mesh.add_plot(plt)
    # plt.show()
    plt.savefig('mesh-1.png')
    plt.close()

    print('end of the file')

