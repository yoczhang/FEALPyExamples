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

from Stokes2DData import Stokes2DData_0, Stokes2DData_1, Stokes2DData_2
import numpy as np
from ShowCls import ShowCls
from StokesHHOModel2d import StokesHHOModel2d
from fealpy.mesh import MeshFactory
from fealpy.mesh import HalfEdgeMesh2d
import matplotlib.pyplot as plt
import datetime

# from fealpy.tools.show import showmultirate, show_error_table
# from fealpy.mesh.mesh_tools import find_entity


# --- begin setting --- #
d = 2  # the dimension
p = 1  # the polynomial order
n = 2  # the number of refine mesh
maxit = 4  # the max iteration of the mesh

nu = 1.0e-0
pde = Stokes2DData_2(nu)  # create pde model

# --- error settings --- #
errorType = ['$|| u - u_h||_0$', '$||\\nabla u - \\nabla u_h||_0$', '|| p - p_h ||_0', 'eta']
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)
Ndof = np.zeros(maxit, dtype=np.int)  # the array to store the number of dofs

# --- mesh setting --- #
box = [0, 1, 0, 1]  # [0, 1]^2 domain
mf = MeshFactory()
meshtype = 'quad'
mesh = mf.boxmesh2d(box, nx=n, ny=n, meshtype=meshtype)
mesh = HalfEdgeMesh2d.from_mesh(mesh)
mesh.init_level_info()
mesh.uniform_refine(n - 1)  # refine the mesh at beginning

now_time = datetime.datetime.now()
outPath = '../Outputs/PostStokes' + now_time.strftime('%y-%m-%d(%H\'%M\'%S)')

# --- plot the mesh --- #
# fig = plt.figure()
# axes = fig.gca()
# mesh.add_plot(axes, cellcolor='w')
# find_entity(axes, mesh, entity='cell', showindex=True, color='b', markersize=10, fontsize=8)
# find_entity(axes, mesh, entity='edge', showindex=True, color='r', markersize=10, fontsize=8)
# find_entity(axes, mesh, entity='node', showindex=True, color='y', markersize=10, fontsize=8)
# plt.show()
sc = ShowCls(p, mesh, errorType=errorType, Ndof=Ndof, errorMatrix=errorMatrix, out=outPath)
# sc.showMeshInfo()
# sc.showMesh()

# mesh.uniform_refine(1)
# print("------------------")
# sc.showMeshInfo()

# --- start for-loop --- #
stokes = None
sol = None
for i in range(maxit):
    print('\n# --------------------- i = %d ------------------------- #' % i)
    stokes = StokesHHOModel2d(pde, mesh, p)
    sol = stokes.solve()
    Ndof[i] = stokes.space.number_of_global_dofs()  # get the number of dofs
    errorMatrix[0, i] = nu * stokes.velocity_L2_error()  # get the velocity L2 error
    errorMatrix[1, i] = nu * stokes.velocity_energy_error()  # get the velocity energy error
    errorMatrix[2, i] = nu ** (-1) * stokes.pressure_L2_error()  # get the pressure L2 error

    # --- adaptive settings --- #
    uh = sol['uh']
    eta = stokes.space.residual_estimate0(nu, uh, pde.source, pde.velocity)
    errorMatrix[3, i] = np.sum(eta)
    eff = np.sqrt(sum(eta) ** 2 / (errorMatrix[1, i] * errorMatrix[1, i] + errorMatrix[2, i] * errorMatrix[2, i]))
    print('Posteriori Info:')
    print('  |___ eff: ', eff)
    print('  |___ before refine: number of cells: ', mesh.number_of_cells())
    # sc.showMesh(markCell=False, markEdge=False, markNode=False)
    fig1 = plt.figure()
    axes = fig1.gca()
    mesh.add_plot(axes)
    outPath_1 = outPath + str(i) + '-mesh.png'
    plt.savefig(outPath_1)
    plt.close()

    # --- refine the mesh --- #
    # aopts = mesh.adaptive_options(method='max', theta=0.3, maxcoarsen=0.1, HB=True)
    # mesh.adaptive(eta, aopts) if i < maxit - 1 else None

    # isMarkedCell = stokes.space.post_estimator_markcell(eta, theta=0.5)
    # mesh.refine_poly(isMarkedCell) if i < maxit - 1 else None

    mesh.uniform_refine() if i < maxit - 1 else None
    print('  |___ after refine: number of cells: ', mesh.number_of_cells())

# --- plot solution --- #
stokes.showSolution(sc)

# --- get the convergence rate --- #
sc.show_error_table()
sc.showmultirate(0)
plt.show()

# ---
print('end of the program')
