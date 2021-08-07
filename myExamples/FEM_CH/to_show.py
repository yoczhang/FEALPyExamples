#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: to_show.py
# @Author: Yongchao Zhang, Northwest University, Xi'an
# @E-mail: yoczhang@nwu.edu.cn
# @Site:
# @Time: Aug 07, 2021
# ---


import numpy as np
from types import ModuleType


def show_error_table(N, errorType, errorMatrix, f='e', pre=4, sep=' & '):

    n = errorMatrix.shape[1] + 1
    print('\\begin{table}[!htdp]')
    print('\\begin{tabular}[c]{|' + n * 'c|' + '}\hline')

    s = 'Dof' + sep + np.array2string(N, separator=sep,)
    s = s.replace('\n', '')
    s = s.replace('[', '')
    s = s.replace(']', '')
    print(s)
    print('\\\\\\hline')

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
        print(s)
        print('\\\\\\hline')

        order = np.log(line[0:-1] / line[1:]) / np.log(2)
        s = 'Order' + sep + '--' + sep + np.array2string(order,
                                                         separator=sep, precision=2)
        s = s.replace('\n', '')
        s = s.replace('[', '')
        s = s.replace(']', '')
        print(s)
        print('\\\\\\hline')

    print('\\end{tabular}')
    print('\\end{table}')

def showmultirate(plot, k, N, errorMatrix, labellist, optionlist=None, lw=1,
                  ms=4, propsize=10):
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
        if len(N.shape) == 1:
            showrate(axes, k, N, errorMatrix[i], optionlist[i], label=labellist[i], lw=lw, ms=ms)
        else:
            showrate(axes, k, N[i], errorMatrix[i], optionlist[i], label=labellist[i], lw=lw, ms=ms)
    axes.legend(loc=3, framealpha=0.2, fancybox=True, prop={'size': propsize})
    return axes


def showrate(axes, k, N, error, option, label=None, lw=1, ms=4):
    axes.set_xlim(left=N[0] / 2, right=N[-1] * 2)
    line0, = axes.loglog(N, error, option, lw=lw, ms=ms, label=label)
    if isinstance(k, int):
        c = np.polyfit(np.log(N[k:]), np.log(error[k:]), 1)
        s = 0.75 * error[k] / N[k] ** c[0]
        line1, = axes.loglog(N[k:], s * N[k:] ** c[0], label='C$N^{%0.4f}$' % (c[0]),
                             lw=lw, ls=line0.get_linestyle(), color=line0.get_color())
    else:
        c = np.polyfit(np.log(N[k]), np.log(error[k]), 1)
        s = 0.75 * error[k[0]] / N[k[0]] ** c[0]
        line1, = axes.loglog(N[k], s * N[k] ** c[0], label='C$N^{%0.4f}$' % (c[0]),
                             lw=lw, ls=line0.get_linestyle(), color=line0.get_color())


def showmultirate1(plot, k, N, errorMatrix, labellist, optionlist=None, lw=1,
                   ms=4, propsize=10):
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
        if len(N.shape) == 1:
            showrate1(axes, k, N, errorMatrix[i], optionlist[i], label=labellist[i], lw=lw, ms=ms)
        else:
            showrate1(axes, k, N[i], errorMatrix[i], optionlist[i], label=labellist[i], lw=lw, ms=ms)
    axes.legend(loc=3, framealpha=0.2, fancybox=True, prop={'size': propsize})
    return axes


def showrate1(axes, k, N, error, option, label=None, lw=1, ms=4):
    #    axes.set_xlim(left=N[0]/2, right=N[-1]*2)
    line0, = axes.semilogy(N, error, option, lw=lw, ms=ms, label=label)


