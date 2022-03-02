#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: loadWaveData.py
# @Author: Yongchao Zhang
# @Institution: Northwest University, Xi'an, Shaanxi, China
# @E-mail: yoczhang@126.com, yoczhang@nwu.edu.cn
# @Site: 
# @Time: Mar 01, 2022
# ---


import numpy as np
import scipy.io as io

filename = './time_position_Xi_20220301-141024'

# |--- load file
load_filename = filename + '.npy'
# wavedata = np.load('./time_position_Xi_20220227-173723.npy')
wavedata = np.load(load_filename)

# |--- save to matlab m-file
print('save to matlab m-file')
save_filename = filename + '.mat'
io.savemat(save_filename, {'wavedata': wavedata})

print('end of the file')
