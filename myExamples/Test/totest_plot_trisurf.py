#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: totest_plot_trisurf.py
# @Author: Yongchao Zhang, Northwest University, Xi'an
# @E-mail: yoczhang@nwu.edu.cn
# @Site:
# @Time: Oct 06, 2020
# ---


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as mtri

fig = plt.figure()
ax = fig.gca(projection='3d')

xy = [[0.3, 0.5],
      [0.6, 0.8],
      [0.5, 0.1],
      [0.1, 0.2]]
xy = np.array(xy)
z = np.zeros(8)

triangles = [[0, 2, 1],
             [2, 0, 3]]

triang = mtri.Triangulation(xy[:, 0], xy[:, 1], triangles=triangles)
plt.triplot(triang, marker="o")

ax.plot_trisurf(triang, z, linewidth=0.2, antialiased=True)

ax.view_init(45, -90)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_aspect("equal")

fig.set_size_inches(8, 8)

plt.show()
