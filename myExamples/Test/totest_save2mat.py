#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: totest_save2mat.py
# @Author: Yongchao Zhang
# @Institution: Northwest University, Xi'an, Shaanxi, China
# @E-mail: yoczhang@126.com, yoczhang@nwu.edu.cn
# @Site: 
# @Time: Sep 06, 2022
# ---


from scipy.io import loadmat, savemat
import numpy as np


surf_x = np.arange(24).reshape(4, 6)
surf_y = surf_x * 0.1
surf_u = surf_x + surf_y

line_x = np.arange(12)
line_u = line_x * 0.2

savemat('LS.mat', {'line': (line_x, line_u), 'surf': [surf_x, surf_y, surf_u]})
# savemat('LS.mat', {'loss': [0.003, 0.342222, 0.005559]}, appendmat=True)


print('end of the file')
