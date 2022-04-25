#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: CoCurrentFlowPlotResult.py
# @Author: Yongchao Zhang, Northwest University, Xi'an
# @E-mail: yoczhang@nwu.edu.cn
# @Site:
# @Time: Apr 25, 2022
# ---

import matplotlib.pyplot as plt
import numpy as np

val0_at_0 = np.load('val0_at_0_20220425-121935.npy')
plt.figure()
plt.plot(val0_at_0[:, 1], val0_at_0[:, 0])
plt.xlabel("time")
plt.ylabel("V")

print('end of the file')
