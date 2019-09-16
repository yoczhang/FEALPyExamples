#!/anaconda3/envs/FEALPy/bin python3.7
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: MainStokesMAC.py
# @Author: Yongchao Zhang
# @E-mail: yoczhang@126.com
# @Time: Sep 11, 2019
# ---


import numpy as np
from PDEdata import StokesMACData
from Smoothers_2D import DGS_smoother

pde = StokesMACData()

uh, vh, ph = DGS_smoother(pde)

print("test end")





