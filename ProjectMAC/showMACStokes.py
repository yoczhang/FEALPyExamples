#!/anaconda3/envs/FEALPy/bin python3.7
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: showMACStokes.py
# @Author: Yongchao Zhang
# @E-mail: yoczhang@126.com
# @Time: Sep 21, 2019
# ---

import sys
import numpy as np
from fealpy.tools.show import show_error_table


# overload the fealpy function: show_error_table
def my_show_error_table(N, errorType, errorMatrix, showTable='No'):
    if showTable is 'No':
        f = 'e'
        pre = 4
        sep = ' & '
        out = sys.stdout
        end = '\n'

        s = 'Dof' + sep + np.array2string(N, separator=sep,
                                          )
        s = s.replace('\n', '')
        s = s.replace('[', '')
        s = s.replace(']', '')
        print(s, file=out, end=end)

        n = len(errorType)
        ff = '%.' + str(pre) + f
        for i in range(n):
            first = errorType[i]
            line = errorMatrix[i]
            s = first + sep + np.array2string(line, separator=sep,
                                              precision=pre, formatter=dict(float=lambda x: ff % x))

            s = s.replace('\n', '')
            s = s.replace('[', '')
            s = s.replace(']', '')
            print(s, file=out, end=end)
    else:
        show_error_table(N, errorType, errorMatrix)

