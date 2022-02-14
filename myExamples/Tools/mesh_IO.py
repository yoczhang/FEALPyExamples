#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: mesh_IO.py
# @Author: Yongchao Zhang, Northwest University, Xi'an
# @E-mail: yoczhang@nwu.edu.cn
# @Site:
# @Time: Nov 01, 2020
# ---

__doc__ = """
这是个很粗糙的 IO 类, 主要目的就是方便转换自己 fealpy 和 MATLAB 中网格. 
"""

from scipy.io import loadmat, savemat
import numpy as np
from fealpy.mesh.PolygonMesh import PolygonMesh
import scipy.io as io
from ShowCls import ShowCls


class mesh_IO:
    def __init__(self):
        # self.filename = filename
        # self.mfile = loadmat(self.filename)
        pass

    def loadMatlabMesh(self, filename=None):
        # filename = self.filename if filename is None else filename
        # mfile = self.mfile if filename is None else loadmat(filename)
        mfile = loadmat(filename)
        pnode = mfile['node']
        pelem = mfile['elem']
        Nelem = pelem.shape[0]
        pcellLocation = np.zeros((Nelem + 1,), dtype=np.int)
        pcell = np.zeros((0,), dtype=np.int)

        for n in range(Nelem):
            lcell = np.squeeze(pelem[n][0]) - 1  # let the index from 0.
            lNedge = lcell.shape[0]
            pcellLocation[n + 1] = pcellLocation[n] + lNedge
            pcell = np.concatenate([pcell, lcell])

        mesh = PolygonMesh(pnode, pcell, pcellLocation)
        return mesh

    def save2MatlabMesh(self, mesh, filename=None):
        node = np.array(mesh.node)
        c2n = mesh.ds.cell_to_node()
        # nvc = mesh.number_of_vertices_of_cells()
        if isinstance(c2n, tuple):
            cell_all = c2n[0] + 1  # 转换为 MATLAB 数据时, 编号从 1 开始.
            cell_location = c2n[1]
            cell = np.split(cell_all, cell_location[1:-1])
        elif isinstance(c2n, np.ndarray):
            cell = c2n + 1
        else:
            raise ValueError("In `save2MatlabMesh()`, the `c2n` type is wrong")
        io.savemat(filename, {'node': node, 'elem': cell})

    def save2MatlabUh(self, Uh, filename=None):
        io.savemat(filename, {'Uh': Uh})




