#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: totest_mapfunc.py
# @Author: Yongchao Zhang, Northwest University, Xi'an
# @E-mail: yoczhang@nwu.edu.cn
# @Site:
# @Time: Oct 06, 2020
# ---


import numpy as np


def square(x):
    print("in square func")
    return x**2


# TODO: map() 必须与 list() 搭配使用? 否则, 不会进入 'square' 函数中?
aa = map(square, [1, 2, 3, 4, 5])

bb = list(map(square, [1, 2, 3, 4, 5]))






