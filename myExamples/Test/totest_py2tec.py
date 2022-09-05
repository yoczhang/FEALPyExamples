#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: totest_py2tec.py
# @Author: Yongchao Zhang, Northwest University, Xi'an
# @E-mail: yoczhang@nwu.edu.cn
# @Site:
# @Time: Sep 04, 2022
# ---


import sys

sys.path.insert(0, '../Tools/')

import py2tec
import numpy as np

tdata = {'varnames': ['x', 'y'],
         'lines': [
             {'zonename': 'line1',
              'data': [np.array([1, 2, 3], np.int32), np.array([-1, -2, -3], np.float64)]}
         ]}

# |--- test1, export to tec
py2tec.py2tec(tdata, './test.tec')

# |--- test2, import from tec
tdata = py2tec.tec2py('./test.tec')


