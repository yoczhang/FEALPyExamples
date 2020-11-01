#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: loadMatlabFile.py
# @Author: Yongchao Zhang, Northwest University, Xi'an
# @E-mail: yoczhang@nwu.edu.cn
# @Site:
# @Time: Nov 01, 2020
# ---


from scipy.io import loadmat
import numpy as np
from fealpy.mesh.PolygonMesh import PolygonMesh
from ShowCls import ShowCls


class loadMatlabFile:
    def __init__(self, filename):
        self.filename = filename
        self.mfile = loadmat(self.filename)

    def loadMatlabMesh(self):
        mfile = self.mfile
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
