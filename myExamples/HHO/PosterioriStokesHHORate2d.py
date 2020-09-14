#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: PosterioriStokesHHORate2d.py
# @Author: Yongchao Zhang, Northwest University, Xi'an
# @E-mail: yoczhang@nwu.edu.cn
# @Site:
# @Time: Sep 14, 2020
# ---

__doc__ = """
The fealpy program for posteriori Stokes problem. 
"""

from Stokes2DData import Stokes2DData_0
import numpy as np
from ShowCls import show
from StokesHHOModel2d import StokesHHOModel2d
from fealpy.mesh import MeshFactory
from fealpy.mesh import HalfEdgeMesh2d


# --- begin setting --- #
d = 2  # the dimension
p = 1  # the polynomial order
n = 2  # the number of refine mesh
maxit = 4  # the max iteration of the mesh

nu = 1.0e-3
pde = Stokes2DData_0(nu)  # create pde model

# --- error settings --- #
errorType = ['$|| u - u_h||_0$', '$||\\nabla u - \\nabla u_h||_0$', '|| p - p_h ||_0']
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)

Ndof = np.zeros(maxit, dtype=np.int)  # the array to store the number of dofs

# --- mesh setting --- #
box = [0, 1, 0, 1]  # [0, 1]^2 domain
mf = MeshFactory()
meshtype = 'quad'
mesh = mf.boxmesh2d(box, nx=n, ny=n, meshtype=meshtype)
mesh = HalfEdgeMesh2d.from_mesh(mesh)
mesh.init_level_info()
mesh.uniform_refine(n)

# --- start for-loop --- #
for i in range(maxit):
    stokes = StokesHHOModel2d(pde, mesh, p)
    sol = stokes.solve()
    Ndof[i] = stokes.space.number_of_global_dofs()  # get the number of dofs
    errorMatrix[0, i] = stokes.velocity_L2_error()  # get the velocity L2 error
    errorMatrix[1, i] = stokes.velocity_energy_error()  # get the velocity energy error
    errorMatrix[2, i] = stokes.pressure_L2_error()  # get the pressure L2 error






