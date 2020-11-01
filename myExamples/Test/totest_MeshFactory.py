#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: totest_MeshFactory.py
# @Author: Yongchao Zhang, Northwest University, Xi'an
# @E-mail: yoczhang@nwu.edu.cn
# @Site:
# @Time: Nov 01, 2020
# ---


import numpy as np
from fealpy.mesh import MeshFactory
from ShowCls import ShowCls
from fealpy.mesh import HalfEdgeMesh2d


# --- mesh setting --- #
n = 2
p = 1
box = [0, 1, 0, 1]  # [0, 1]^2 domain
mf = MeshFactory()
meshtype = 'poly'
mesh = mf.boxmesh2d(box, nx=n, ny=n, meshtype=meshtype)
# mesh = mf.special_boxmesh2d(box)
# mesh = HalfEdgeMesh2d.from_mesh(mesh)
# mesh.init_level_info()

sc = ShowCls(p, mesh)
sc.showMesh()

