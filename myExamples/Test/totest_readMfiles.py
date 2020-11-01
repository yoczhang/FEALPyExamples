#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: totest_readMfiles.py
# @Author: Yongchao Zhang, Northwest University, Xi'an
# @E-mail: yoczhang@nwu.edu.cn
# @Site:
# @Time: Nov 01, 2020
# ---


from scipy.io import loadmat
import numpy as np
from fealpy.mesh.PolygonMesh import PolygonMesh
from ShowCls import ShowCls

m = loadmat('../Meshfiles/Dmesh_nonconvex_[0,1]x[0,1]_4.mat')
pnode = m['node']
pelem = m['elem']
Nelem = pelem.shape[0]
pcellLocation = np.zeros((Nelem + 1,), dtype=np.int)
pcell = np.zeros((0,), dtype=np.int)

for n in range(Nelem):
    lcell = np.squeeze(pelem[n][0]) - 1  # let the index from 0.
    lNedge = lcell.shape[0]
    pcellLocation[n + 1] = pcellLocation[n] + lNedge
    pcell = np.concatenate([pcell, lcell])
    # print('pcell = ', pcell)

mesh = PolygonMesh(pnode, pcell, pcellLocation)
sc = ShowCls(1, mesh)
sc.showMesh()

# ---
print("End of this file")

