#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: FEMSAV_NavierStokes.py
# @Author: Yongchao Zhang, Northwest University, Xi'an
# @E-mail: yoczhang@nwu.edu.cn
# @Site:
# @Time: May 31, 2021
# ---

__doc__ = """
The fealpy-FEM program for Navier-Stokes problem. 
"""

import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh import MeshFactory as MF

box = [0, 1, 0, 1]

mesh = MF.boxmesh2d(box, nx=10, ny=10, meshtype='tri')
mesh = MF.boxmesh2d(box, nx=10, ny=10, meshtype='quad')
mesh = MF.boxmesh2d(box, nx=10, ny=10, meshtype='poly')

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
plt.show()