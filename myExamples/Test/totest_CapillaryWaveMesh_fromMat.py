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


mfile = loadmat('CapillaryWaveMesh.mat')
node = mfile['node']
cell = mfile['elem']
mesh = TriangleMesh(node, cell)
mesh = HalfEdgeMesh2d.from_mesh(mesh, NV=3)


