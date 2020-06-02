#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: NavierStokesHHORate2d.py
# @Author: Yongchao Zhang
# @E-mail: yoczhang@126.com
# @Time: May 29, 2020
# ---


from NavierStokes2DData import NavierStokes2DData_0
import numpy as np
import matplotlib.pyplot as plt
from fealpy.tools.show import showmultirate, show_error_table
from StokesHHOModel2d import StokesHHOModel2d
from fealpy.mesh.mesh_tools import find_entity


# --- begin setting --- #
d = 2  # the dimension
p = 1  # the polynomial order
n = 2  # the number of refine mesh
maxit = 4  # the max iteration of the mesh

pde = NavierStokes2DData_0()  # create pde model

# # error settings
errorType = ['$|| u - u_h||_0$', '$||\\nabla u - \\nabla u_h||_0$', '|| p - p_h ||_0']
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)

Ndof = np.zeros(maxit, dtype=np.int)  # the array to store the number of dofs




