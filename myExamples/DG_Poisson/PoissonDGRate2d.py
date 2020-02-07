#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: PoissonDGRate2d.py
# @Author: Yongchao Zhang
# @E-mail: yoczhang@126.com
# @Time: Jan 31, 2020
# ---


from poisson2DData import CosCosData as PDE
import numpy as np
import matplotlib.pyplot as plt
from fealpy.tools.show import showmultirate, show_error_table
from PoissonDGModel2d import PoissonDGModel2d

import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)
# sys.path.append("/Users/yczhang/Documents/FEALPy/FEALPyExamples/myExamples/DG_Poisson")


# --- begin setting --- #
d = 2  # the dimension
p = 1  # the polynomial order
n = 2  # the number of refine mesh
maxit = 3  # the max iteration of the mesh

pde = PDE()  # create pde model
pde.epsilon = -1  # setting the DG-scheme parameter
# # epsilon maybe take -1, 0, 1,
# # the corresponding DG-scheme is called symmetric interior penalty Galerkin (SIPG),
# # incomplete interior penalty Galerkin (IIPG) and nonsymmetric interior penalty Galerkin (NIPG)

pde.eta = 9  # setting the penalty parameter
# # eta may change corresponding the polynomial order 'p' and 'epsilon'

# # error settings
errorType = ['$|| u - u_h||_0$', '$||\\nabla u - \\nabla u_h||_0$']
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)

Ndof = np.zeros(maxit, dtype=np.int)  # the array to store the number of dofs

qtree = pde.init_mesh(n, meshtype='quadtree')
mesh = qtree.to_pmesh()
# ---- poly mesh ----
# h = 0.2
# box = [0, 1, 0, 1]  # [0, 1]^2 domain
# mesh = triangle(box, h, meshtype='polygon')
# -------------------
# # TODO: need to test the 'tri'-mesh and give more simple way to construct polygon mesh

for i in range(maxit):
    dg = PoissonDGModel2d(pde, mesh, p, q=p+2)
    ls = dg.solve()
    Ndof[i] = dg.space.number_of_global_dofs()  # 获得空间自由度个数
    errorMatrix[0, i] = dg.L2_error()  # 计算 L2 误差
    errorMatrix[1, i] = dg.H1_semi_error()  # 计算 H1 误差
    if i < maxit - 1:
        mesh.uniform_refine()  # 一致加密网格


