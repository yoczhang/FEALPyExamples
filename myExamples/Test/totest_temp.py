#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: totest_temp.py
# @Author: Yongchao Zhang
# @E-mail: yoczhang@126.com
# @Time: Apr 28, 2020
# ---

import numpy as np

a1 = np.array((1, 2, 3))
a2 = np.array((4, 5, 6))
aa = [a1, a2]

bb = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]])

cc = 0


def sum_t(x):
    r = x[0] + x[1]

    global cc
    cc += sum(r)
    # return r


# map(sum_t, zip(aa, bb))
dd = list(map(sum_t, zip(aa, bb)))

# --- another test --- #
z1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
z2 = np.array([[0.1, 0.2], [0.3, 0.4]])
rowIndx = np.array([[0, 0], [1, 1]])
colIndx = np.array([[0, 1], [0, 1]])

# zz = csr_matrix((z2.flat, (rowIndx.flat, colIndx.flat)), shape=(5, 5))

A = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]])
b = np.array([[1, 2, 3], [4, 5, 6]])
x = np.einsum('ijk, ik->ij', A, b)

# ------------------------------------------------- #
print("End of this test file")


# old api
def integral(self, u, celltype=False, barycenter=True):
    """
        """
    qf = self.integrator
    bcs = qf.quadpts  # 积分点 (NQ, 3)
    ws = qf.weights  # 积分点对应的权重 (NQ, )
    if barycenter or u.coordtype == 'barycentric':
        val = u(bcs)
    else:
        ps = self.mesh.bc_to_point(bcs)  # (NQ, NC, 2)
        val = u(ps)
    dim = len(ws.shape)
    s0 = 'abcde'
    s1 = '{}, {}j..., j->j...'.format(s0[0:dim], s0[0:dim])
    e = np.einsum(s1, ws, val, self.cellmeasure)
    if celltype is True:
        return e
    else:
        return e.sum()
