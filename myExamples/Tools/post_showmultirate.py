#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: post_showmultirate.py
# @Author: Yongchao Zhang, Northwest University, Xi'an
# @E-mail: yoczhang@nwu.edu.cn
# @Site:
# @Time: Nov 28, 2020
# ---


import numpy as np
from ShowCls import ShowCls
from fealpy.mesh import MeshFactory

# --- mesh1 --- #
p = 1
n = 2
box = [0, 1, 0, 1]  # [0, 1]^2 domain
mf = MeshFactory()
meshtype = 'quad'
mesh = mf.boxmesh2d(box, nx=n, ny=n, meshtype=meshtype)  # 这里 mesh 只是用来初始化

# --- setting the errorType, Ndof, errorMatrix
errorType = ['$\\bf{e_u}$', 'eta']
Ndof = np.array(
    [1156, 1220, 1284, 1384, 1440, 1540, 1632, 1834, 2138, 2542, 3114, 3850, 4914, 6402, 8200, 10628, 13694, 17504, 22432, 28786,
     37010, 47090, 59662, 75838, 97126], dtype=np.int)
errorMatrix = np.array([[1.1555e+00, 9.8976e-01, 8.5203e-01, 7.0781e-01, 6.4650e-01, 5.6002e-01, 4.9564e-01, 4.1141e-01,
                         3.3548e-01, 2.7535e-01, 2.2500e-01, 1.7799e-01, 1.4232e-01, 1.1417e-01, 9.0159e-02, 7.1493e-02,
                         5.6420e-02, 4.4774e-02, 3.5566e-02, 2.8458e-02, 2.2695e-02, 1.8067e-02, 1.4443e-02, 1.1481e-02,
                         9.1255e-03],
                        [8.3309e-01, 7.1849e-01, 6.3336e-01, 6.1243e-01, 5.2297e-01, 5.1540e-01, 4.5707e-01, 3.9608e-01,
                         3.3198e-01, 2.7814e-01, 2.2982e-01, 1.8185e-01, 1.4623e-01, 1.1684e-01, 9.2360e-02, 7.3563e-02,
                         5.8168e-02, 4.5930e-02, 3.6529e-02, 2.9112e-02, 2.2914e-02, 1.8181e-02, 1.4442e-02, 1.1451e-02,
                         9.0836e-03]],
                       dtype=np.float)

sc = ShowCls(p, mesh, errorType=errorType, Ndof=Ndof, errorMatrix=errorMatrix, out=None)
plt = sc.showmultirate(len(Ndof)-7)
# plt.xlabel('Number of unknowns')
# plt.ylabel('Errors')

print('--- end of the code ---')


