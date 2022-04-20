#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: RefineCoCurrentFlowMesh.py
# @Author: Yongchao Zhang
# @Institution: Northwest University, Xi'an, Shaanxi, China
# @E-mail: yoczhang@126.com, yoczhang@nwu.edu.cn
# @Site: 
# @Time: Apr 20, 2022
# ---


import numpy as np
from scipy.io import loadmat
from fealpy.mesh.TriangleMesh import TriangleMesh
from fealpy.mesh.HalfEdgeMesh2d import HalfEdgeMesh2d
import matplotlib  # 为了解决画图时采用 GUI (plt.show()) 的形式时, python3.8 崩溃的情况.
matplotlib.use("TkAgg")  # 为了解决画图时采用 GUI (plt.show()) 的形式时, python3.8 崩溃的情况.
import matplotlib.pyplot as plt

meshname = 'CoCurrentFlowMesh'
mfile = loadmat('./' + meshname + '.mat')
node = mfile['node']
cell = mfile['elem']
mesh = TriangleMesh(node, cell)
mesh = HalfEdgeMesh2d.from_mesh(mesh, NV=3)

# refine the mesh at the interface
NC = mesh.number_of_cells()
cellstart = mesh.ds.cellstart
isMarkedCell = np.zeros(NC + cellstart, dtype=np.bool_)
isMarkedCell[cellstart:] = True
mesh.refine_triangle_nvb(isMarkedCell)
node = np.array(mesh.node)
cell = mesh.ds.cell_to_node()
mesh = TriangleMesh(node, cell)
mesh = HalfEdgeMesh2d.from_mesh(mesh, NV=3)

# another refine
bc = mesh.cell_barycenter()
NC = mesh.number_of_cells()
cellstart = mesh.ds.cellstart
isMarkedCell = np.zeros(NC + cellstart, dtype=np.bool_)
isMarkedCell[cellstart:] = abs(abs(bc[:, 1]) - 0.5) <= 0.1
mesh.refine_triangle_rg(isMarkedCell)

# left_node, = np.nonzero(abs(node[:, 0]-0) < 1e-9)
# right_node, = np.nonzero(abs(node[:, 0]-0.8) < 1e-9)
# aa = node[left_node, 1] - node[right_node, 1]

# fig = plt.figure()
# axes = fig.gca()
# mesh.add_plot(axes)
# # mesh.find_cell(axes, showindex=True)
# plt.show()
# plt.close()

node = mesh.node
cell = mesh.ds.cell_to_node()
np.save('./CoCurrentMeshNode', node)
np.save('./CoCurrentMeshCell', cell)

print('Mesh-name = %s,  ||  Number-of-mesh-cells = %d' % (meshname, mesh.number_of_cells()))
print('# --------------------------------------------------------------------- #')


