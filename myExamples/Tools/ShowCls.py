#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: ShowCls.py
# @Author: Yongchao Zhang
# @Institution: Northwest University, Xi'an, Shaanxi, China
# @E-mail: yoczhang@126.com, yoczhang@nwu.edu.cn
# @Site: 
# @Time: Aug 06, 2020
# ---


import sys
import numpy as np
from types import ModuleType


class show:
    def __init__(self, plot, meshtype, GD, k_slope, errortypeList, NdofList, errorMatrix):
        self.plot = plot
        self.meshtype = meshtype
        self.GD = GD
        self.k_slope = k_slope
        self.errortypeList = errortypeList
        self.NdofList = NdofList
        self.errorMatrix = errorMatrix

    def show_error_table(self, f='e', pre=4, sep=' & ', out=sys.stdout, end='\n'):
        GD = self.GD
        meshtype = self.meshtype
        NdofList = self.NdofList
        errortypeList = self.errortypeList
        errorMatrix = self.errorMatrix

        hh = np.power(1/NdofList, 1/GD)

        flag = False
        if type(out) == type(''):
            flag = True
            out = open(out, 'w')

        n = errorMatrix.shape[1] + 1
        print('\\begin{table}[!htdp]', file=out, end='\n')
        print('\\begin{tabular}[c]{|' + n * 'c|' + '}\hline', file=out, end='\n')

        s = 'Dof' + sep + np.array2string(NdofList, separator=sep, )
        s = s.replace('\n', '')
        s = s.replace('[', '')
        s = s.replace(']', '')
        print(s, file=out, end=end)
        print('\\\\\\hline', file=out)

        n = len(errortypeList)
        ff = '%.' + str(pre) + f
        for i in range(n):
            first = errortypeList[i]
            line = errorMatrix[i]
            s = first + sep + np.array2string(line, separator=sep,
                                              precision=pre, formatter=dict(float=lambda x: ff % x))

            s = s.replace('\n', '')
            s = s.replace('[', '')
            s = s.replace(']', '')
            print(s, file=out, end=end)
            print('\\\\\\hline', file=out)

            if meshtype == 'tri':
                order = np.log(line[0:-1] / line[1:]) / np.log(2)
            else:
                order = np.log(line[0:-1] / line[1:]) / np.log(hh[0:-1]/hh[1:])
            s = 'Order' + sep + '--' + sep + np.array2string(order, separator=sep, precision=2)
            s = s.replace('\n', '')
            s = s.replace('[', '')
            s = s.replace(']', '')
            print(s, file=out, end=end)
            print('\\\\\\hline', file=out)

        print('\\end{tabular}', file=out, end='\n')
        print('\\end{table}', file=out, end='\n')

        if flag:
            out.close()

    def showmultirate(self, optionlist=None, lw=1, ms=4, propsize=10):
        plot = self.plot
        k_slope = self.k_slope
        NdofList = self.NdofList
        errorMatrix = self.errorMatrix
        errortypeList = self.errortypeList
        if isinstance(plot, ModuleType):
            fig = plot.figure()
            fig.set_facecolor('white')
            axes = fig.gca()
        else:
            axes = plot
        if optionlist is None:
            optionlist = ['k-*', 'r-o', 'b-D', 'g-->', 'k--8', 'm--x', 'r-.x', 'b-.+', 'b-.h', 'm:s', 'm:p', 'm:h']

        m, n = errorMatrix.shape
        for i in range(m):
            if len(NdofList.shape) == 1:
                self.showrate(axes, k_slope, NdofList, errorMatrix[i], optionlist[i], label=errortypeList[i], lw=lw, ms=ms)
            else:
                self.showrate(axes, k_slope, NdofList[i], errorMatrix[i], optionlist[i], label=errortypeList[i], lw=lw, ms=ms)
        axes.legend(loc=3, framealpha=0.2, fancybox=True, prop={'size': propsize})
        return axes

    def showrate(self, axes, k, N, error, option, label=None, lw=1, ms=4):
        axes.set_xlim(left=N[0] / 2, right=N[-1] * 2)
        line0, = axes.loglog(N, error, option, lw=lw, ms=ms, label=label)
        if isinstance(k, int):
            c = np.polyfit(np.log(N[k:]), np.log(error[k:]), 1)  # 目的是从第 k 个取值来计算斜率, c[0] 即斜率
            s = 0.75 * error[k] / N[k] ** c[0]  # 在第 k 个值处, 固定直线和折线的距离, 直线是给定的类似基准线, 折线就是误差的斜率线
            line1, = axes.loglog(N[0:], s * N[0:] ** c[0], label='C$N^{%0.4f}$' % (c[0]),
                                 lw=lw, ls=line0.get_linestyle(), color=line0.get_color())
            # # 如果 line1,=axes.loglog(N[k:],s*N[k:]**c[0], ...) 则表示从第 k 个值处开始画直线, 一直画到最后一个值
        else:
            c = np.polyfit(np.log(N[k]), np.log(error[k]), 1)
            s = 0.75 * error[k[0]] / N[k[0]] ** c[0]
            line1, = axes.loglog(N[k], s * N[k] ** c[0], label='C$N^{%0.4f}$' % (c[0]),
                                 lw=lw, ls=line0.get_linestyle(), color=line0.get_color())

