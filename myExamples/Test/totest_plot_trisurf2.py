#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: totest_plot_trisurf2.py
# @Author: Yongchao Zhang, Northwest University, Xi'an
# @E-mail: yoczhang@nwu.edu.cn
# @Site:
# @Time: Oct 07, 2020
# ---

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as mtri

xy = [[0.3, 0.5],
      [0.6, 0.8],
      [0.5, 0.1],
      [0.1, 0.2]]
xy = np.array(xy)

triangles = np.array([[0, 2, 1], [2, 0, 3]])

triang = mtri.Triangulation(xy[:, 0], xy[:, 1])
# plt.triplot(triang, marker="o")

x = xy[:, 0]
y = xy[:, 1]
z = np.zeros(x.shape)
fig = plt.figure()
ax = fig.gca(projection='3d')
# ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.2)
ax.plot_trisurf(triang, z, linewidth=0.2, antialiased=True)

plt.show()
