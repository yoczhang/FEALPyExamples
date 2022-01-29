#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: totest_meshrefine.py
# @Author: Yongchao Zhang
# @Institution: Northwest University, Xi'an, Shaanxi, China
# @E-mail: yoczhang@126.com, yoczhang@nwu.edu.cn
# @Site: 
# @Time: Jan 29, 2022
# ---

import numpy as np
import sys
import time
from fealpy.mesh.HalfEdgeMesh2d import HalfEdgeMesh2d
from fealpy.mesh import Quadtree
from fealpy.mesh import TriangleMesh, PolygonMesh, QuadrangleMesh
import matplotlib  # 为了解决画图时采用 GUI (plt.show()) 的形式时, python3.8 崩溃的情况.
matplotlib.use("TkAgg")  # 为了解决画图时采用 GUI (plt.show()) 的形式时, python3.8 崩溃的情况.
import matplotlib.pyplot as plt