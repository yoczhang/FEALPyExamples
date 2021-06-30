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

# --- test ending --- #
print('end of the test')
