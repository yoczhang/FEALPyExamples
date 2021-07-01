#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: totest_pyfftw.py
# @Author: Yongchao Zhang, Northwest University, Xi'an
# @E-mail: yoczhang@nwu.edu.cn
# @Site:
# @Time: Jun 30, 2021
# ---


import pyfftw as pw

import numpy as np

# --- test1
a = pw.empty_aligned(128, dtype='complex128', n=16)

a[:] = np.random.randn(128) + 1j*np.random.randn(128)

b = pw.interfaces.numpy_fft.fft(a)

c = np.fft.fft(a)

rr = np.allclose(b, c)

print(rr)

# --- test2
N = 8
L = 2*np.pi
h = L/N
x = h*np.arange(N)
fh = np.sin(x)
fh_hat = pw.empty_aligned(N, dtype='complex128', n=16)
fh_hat[:] = pw.interfaces.numpy_fft.fft(fh)
ifh_hat = pw.interfaces.numpy_fft.ifftn(fh_hat)

dfh = np.cos(x)
K = 1j*np.concatenate([np.arange(0, N/2+1), np.arange(-N/2+1, 0)])
dfh_hat = K*fh_hat

inv_dfh_hat = np.real(pw.interfaces.numpy_fft.ifftn(dfh_hat))

rrdfh = np.allclose(dfh, inv_dfh_hat)
print('rrdfh = ', rrdfh)

# --- test ending --- #
print('end of the test')
