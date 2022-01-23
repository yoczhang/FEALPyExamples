#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: totest_HalfEdgeMesh2d.py
# @Author: Yongchao Zhang
# @Institution: Northwest University, Xi'an, Shaanxi, China
# @E-mail: yoczhang@126.com, yoczhang@nwu.edu.cn
# @Site: 
# @Time: Jan 22, 2022
# ---
import numpy as np
from fealpy.mesh.HalfEdgeMesh2d import HalfEdgeMesh2d
from fealpy.mesh import MeshFactory as MF
import matplotlib.pyplot as plt
from fealpy.mesh.mesh_tools import find_node, find_entity

domain = [0, 1, 0, 1]
n = 2
mesh = MF.boxmesh2d(domain, nx=n, ny=n, meshtype='tri')
mesh = HalfEdgeMesh2d.from_mesh(mesh, NV=3)

# fig = plt.figure()
# axes = fig.gca()
# mesh.add_plot(axes, cellcolor='w')
# mesh.find_cell(axes, showindex=True)
# mesh.find_node(axes, node, showindex=True, fontsize=12, markersize=25)
# plt.show()

c2c = mesh.ds.cell_to_cell().todense()


print('end of the file')
