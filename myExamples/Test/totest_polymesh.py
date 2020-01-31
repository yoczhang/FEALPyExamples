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
find_entity(axes, mesh, entity='cell', index='all', showindex=True, color='b', markersize=16, fontsize=9)
find_entity(axes, mesh, entity='node', index='all', showindex=True, color='r', markersize=16, fontsize=9)
# find_node(axes, mesh.node, showindex=True, fontsize=12, markersize=25)
plt.show()


