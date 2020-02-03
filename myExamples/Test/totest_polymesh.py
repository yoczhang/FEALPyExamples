#!/anaconda3/envs/FEALPy/bin python3.7
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: YC_test_polymesh.py
# @Author: Yongchao Zhang
# @E-mail: yoczhang@126.com
# @Time: Aug 19, 2019
# ---

import numpy as np

from fealpy.pde.poisson_model_2d import CrackData, LShapeRSinData, CosCosData, KelloggData, SinSinData
from fealpy.vem import PoissonCVEMModel
from fealpy.tools.show import showmultirate
from fealpy.mesh.simple_mesh_generator import triangle
from fealpy.mesh.mesh_tools import find_node, find_entity

import matplotlib.pyplot as plt

h = 0.2
box = [0, 1, 0, 1]  # [0, 1]^2 domain
mesh = triangle(box, h, meshtype='polygon')

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes, cellcolor='w')
find_entity(axes, mesh, entity='cell', index='all', showindex=True, color='b', markersize=10, fontsize=8)
find_entity(axes, mesh, entity='edge', index='all', showindex=True, color='r', markersize=10, fontsize=8)
find_entity(axes, mesh, entity='node', index='all', showindex=True, color='y', markersize=10, fontsize=8)
# find_entity(axes, mesh, entity='node', index='all', showindex=True, color='r', markersize=16, fontsize=9)
# find_node(axes, mesh.node, showindex=True, fontsize=12, markersize=25)
plt.show()


# -------------------
edge2cell = mesh.ds.edge_to_cell()
nm = mesh.edge_normal()  # The length of the normal-vector isn't 1, is the length of corresponding edge.

# -------------------
# find the nodes of the cells
cells = mesh.ds.cell
cellL = mesh.ds.cellLocation
cell_0 = cells[cellL[0]:cellL[1]]  # the vertices-index of 0-th cell
cell_15 = cells[cellL[15]:cellL[16]]  # the vertices-index of 15-th cell


# ------------------------------------------------- #
print("End of this test file")


