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

from Stokes2DData import Stokes2DData_0, Stokes2DData_1, Stokes2DData_2, Stokes2DData_3, StokesLshapeData
import numpy as np
from ShowCls import ShowCls
from StokesHHOModel2d import StokesHHOModel2d
from fealpy.mesh import MeshFactory as mf
from fealpy.mesh import HalfEdgeMesh2d
from mesh_IO import mesh_IO
import matplotlib.pyplot as plt
import datetime

# from fealpy.tools.show import showmultirate, show_error_table
# from fealpy.mesh.mesh_tools import find_entity


# --- begin setting --- #
d = 2  # the dimension
p = 1  # the polynomial order
n = 4  # the number of refine mesh
maxit = 5  # the max iteration of the mesh

nu = 1.0e-0
pde = Stokes2DData_1(nu)  # create pde model

# --- error settings --- #
errorType = ['$|| u - u_h||_0$', '$||\\nabla u - \\nabla u_h||_0 + s(uh,uh)$', '$|| u - u_h||_{E}$', '|| p - p_h ||_0', 'eta0']
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)
Ndof = np.zeros(maxit, dtype=np.int)  # the array to store the number of dofs

# --- mesh setting --- #
mIO = mesh_IO()

# --- mesh1 --- #
box = [0, 1, 0, 1]  # [0, 1]^2 domain
# meshtype = 'quad'
# mesh = mf.boxmesh2d(box, nx=n, ny=n, meshtype=meshtype)

# --- mesh2 --- #
# matfile = '../Meshfiles/Dmesh_contortedDualTri_[0,1]x[0,1]_4.mat'
# mesh = mIO.loadMatlabMesh(filename=matfile)

# --- mesh3 --- #
# mesh = mf.triangle(box, 1./4)

# --- mesh4: L-shape --- #
matfile = '../Meshfiles/Lshape3_poly_64.mat'
mesh = mIO.loadMatlabMesh(filename=matfile)

# --- to halfedgemesh --- #
mesh = HalfEdgeMesh2d.from_mesh(mesh)
mesh.init_level_info()
# mesh.uniform_refine(n - 1)  # refine the mesh at beginning

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
sc = ShowCls(p, mesh, errorType=errorType, Ndof=Ndof, errorMatrix=errorMatrix, out=None)
# sc.showMeshInfo()
# sc.showMesh()

# mesh.uniform_refine(1)
# print("------------------")
# sc.showMeshInfo()

# --- start for-loop --- #
stokes = None
sol = None
tol = 1.0e-2
print('nu = %e' % nu)
i = 0
ETA = 1.0
uh = None
for i in range(maxit):  # range(maxit), [maxit-1]
    print('\n# --------------------- i = %d ------------------------- #' % i)
    stokes = StokesHHOModel2d(pde, mesh, p)
    sol = stokes.solve()
    uh = sol['uh']

    # --- expand Ndof and errorMatrix --- #
    Ndof[i] = stokes.space.number_of_velocity_dofs()  # get the number of dofs
    errorMatrix[0, i] = (nu ** 0.5) * np.sqrt(np.sum(stokes.velocity_L2_error(celltype=True)**2))  # get the velocity L2 error

    u_post_energyerr = stokes.space.posterror_enengyerror(nu, pde.grad, uh)
    errorMatrix[1, i] = np.sqrt(np.sum(u_post_energyerr**2))  # get the velocity energy error
    errorMatrix[2, i] = (nu ** 0.5) * stokes.velocity_energy_error()  # get the velocity energy error
    errorMatrix[3, i] = (nu ** (-0.5)) * np.sqrt(np.sum(stokes.pressure_L2_error(celltype=True)**2))  # get the pressure L2 error

    # --- adaptive settings --- #
    eta = stokes.space.residual_estimate0(nu, uh, pde.source, pde.velocity)  # (NC,)
    ETA = np.sqrt(np.sum(eta ** 2))
    errorMatrix[4, i] = ETA
    eff0 = 1./np.sqrt(ETA / (errorMatrix[1, i]**2 + errorMatrix[3, i]**2))
    eff1 = 1. / np.sqrt(ETA / (errorMatrix[2, i] ** 2 + errorMatrix[3, i] ** 2))
    print('Posteriori Info:')
    print('  |___ ETA = %e, eff0 = %f, eff1 = %f: ' % (ETA, eff0, eff1))
    print('  |___ before refine: number of cells: %d, pressure dofs: %d ' % (
        mesh.number_of_cells(), stokes.space.number_of_pressure_dofs()))

    # --- refine the mesh --- #
    if i < maxit - 1:
        # --- one way to refine
        # aopts = mesh.adaptive_options(method='max', theta=0.3, maxcoarsen=0.1, HB=True)
        # mesh.adaptive(eta, aopts)

        # --- another way to refine
        # isMarkedCell = stokes.space.post_estimator_markcell(eta, theta=0.3)
        # mesh.refine_poly(isMarkedCell)

        # --- uniform refine the mesh --- #
        mesh.uniform_refine()
    else:
        pass

    # # sc.showMesh(markCell=False, markEdge=False, markNode=False)
    # fig1 = plt.figure()
    # axes = fig1.gca()
    # mesh.add_plot(axes)
    # outPath_1 = outPath + str(i) + '-mesh.png'
    # plt.savefig(outPath_1)
    # plt.close()
    print('  |___ after refine: number of cells: ', mesh.number_of_cells())

# --- plot solution --- #
# stokes.showSolution(sc)

# --- save mesh --- #
# saveMeshName = outPath + '_p=' + str(p) + '_mesh_final.mat'
# mIO.save2MatlabMesh(mesh, filename=saveMeshName)

# --- save uh --- #
# saveUhName = outPath + '_p=' + str(p) + '_uh_final.mat'
# mIO.save2MatlabUh(uh, filename=saveUhName)

# --- get the convergence rate --- #
print('\n')
print('# --------------------- table ------------------------- #')
sc.show_error_table(out=outPath, Cidx=range(i+1), DofName='Velocity-Dof', tableType='dof-type', outFlag=False)
# sc.showmultirate(i-7, Ridx=[1, 4], Cidx=range(i+1), outFlag=False)
# plt.show()

# ---
print('end of the program')
