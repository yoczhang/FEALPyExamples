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
import scipy.io as io
from scipy.special import comb, perm
import matplotlib.pyplot as plt
from fealpy.mesh.mesh_tools import find_entity
# from types import ModuleType


class ShowCls:
    def __init__(self, p, mesh, errorType=None, Ndof=None, errorMatrix=None, out=sys.stdout):
        self.p = p
        self.mesh = mesh
        self.errorType = errorType
        self.Ndof = Ndof
        self.errorMatrix = errorMatrix
        self.out = out

    def showMesh(self, markCell=True, markEdge=True, markNode=True):
        mesh = self.mesh
        fig = plt.figure()
        axes = fig.gca()
        mesh.add_plot(axes, cellcolor='w')
        if markCell:
            find_entity(axes, mesh, entity='cell', showindex=True, color='b', markersize=10, fontsize=8)
        if markEdge:
            find_entity(axes, mesh, entity='edge', showindex=True, color='r', markersize=10, fontsize=8)
        if markNode:
            find_entity(axes, mesh, entity='node', showindex=True, color='y', markersize=10, fontsize=8)
        plt.show()
        plt.close()

    def showMeshInfo(self, out=None, outFlag=True, onlywrite=False):
        p = self.p
        mesh = self.mesh
        out = self.out if out is None else out
        GD = mesh.geo_dimension()
        NC = mesh.number_of_cells()
        NE = mesh.number_of_edges()
        eldof = int(comb(p + GD - 1, GD - 1))
        smsldof = int(comb(p + GD, GD))

        egdof = NE * eldof
        smsgdof = NC * smsldof

        s0 = 'Mesh and Dof info:'
        s1 = '  |___ Polynomial order: ' + str(p) + '.'
        s2 = '  |___ Number of cells: ' + str(NC) + ';  Number of edges: ' + str(NE) + '.'
        s3 = '  |___ Global edge-dofs: ' + str(egdof) + ';  Global smspace cell-dofs: ' + str(smsgdof) + '.'

        if onlywrite is False:
            print(s0)
            print(s1)
            print(s2)
            print(s3)

        flag = False
        outPath = None
        if isinstance(out, str) & outFlag:
            flag = True
            outPath = out if ('.' in out) else out + '_MeshInfo.txt'
            outPath = open(outPath, 'a+')
            print(s0, file=outPath, end='\n')
            print(s1, file=outPath, end='\n')
            print(s2, file=outPath, end='\n')
            print(s3, file=outPath, end='\n')
            print('\n\n------------------------------------------------', file=outPath, end='\n\n')

        if flag:
            outPath.close()

    def showSolution(self, space=None, uh=None, outFlag=True):
        mesh = self.mesh
        out = self.out
        node = mesh.node
        bc = mesh.cell_barycenter()
        edge = mesh.entity('edge')
        edge2cell = mesh.ds.edge_to_cell()

        # --- divide the poly-elems into tri-elems --- #
        Ntri = sum(mesh.number_of_edges_of_cells())
        maskTri = np.arange(Ntri * 3).reshape(-1, 3)

        triCoord0 = np.array([bc[edge2cell[:, 0]], node[edge[:, 0]], node[edge[:, 1]]])  # (3,NE,2)
        # triCoord0 = triCoord0.swapaxes(0, 1)  # (NE,3,2)
        triCoord0 = triCoord0.swapaxes(0, 1).reshape(-1, 2)  # (3*NE,2), each 3-lines is one triangle
        cell0 = np.repeat(edge2cell[:, 0], 3)  # (3*NE,)

        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
        triCoord1 = np.array([bc[edge2cell[isInEdge, 1]], node[edge[isInEdge, 1]], node[edge[isInEdge, 0]]])  # (3,NInE,2)
        triCoord1 = triCoord1.swapaxes(0, 1).reshape(-1, 2)  # (3*NInE,2), each 3-lines is one triangle
        cell1 = np.repeat(edge2cell[isInEdge, 1], 3)  # (3*NInE,)

        triCoord = np.concatenate([triCoord0, triCoord1], axis=0)
        cellIdx = np.concatenate([cell0, cell1])

        triValue = space.value(uh, triCoord, cellIdx)

        # --- save data --- #
        if isinstance(out, str) & outFlag:
            outmat = out + '_plotSolData.mat'
            io.savemat(outmat, {'maskTri': maskTri, 'triCoord': triCoord, 'triValue': triValue})

        # --- plot solution --- #
        fig0 = plt.figure()
        fig0.set_facecolor('white')
        axes = fig0.gca(projection='3d')
        axes.plot_trisurf(triCoord[:, 0], triCoord[:, 1], maskTri, triValue, cmap=plt.get_cmap('jet'), lw=0.0)
        # axes.set_zlim(-2, 2)  # 设置图像z轴的显示范围，x、y轴设置方式相同
        plt.savefig(out + '_Solution.png') if (isinstance(out, str) & outFlag) else None
        plt.close()

    def show_error_table(self, out=None, DofName='Dof', f='e', pre=4, sep=' & ', end='\n', outFlag=True):
        GD = self.mesh.geo_dimension()
        meshtype = self.mesh.meshtype
        Ndof = self.Ndof
        errorType = self.errorType
        errorMatrix = self.errorMatrix
        out = self.out if out is None else out
        hh = Ndof**(-1./GD)

        flag = False
        outPather = None
        if isinstance(out, str) & outFlag:
            flag = True
            outPath = out + '_errtable.txt'
            self.showMeshInfo(out=outPath, onlywrite=True)
            outPather = open(outPath, 'a+')

        n = errorMatrix.shape[1] + 1
        print('\\begin{table}[!htdp]', file=outPather, end='\n')
        print('\\begin{tabular}[c]{|' + n * 'c|' + '}\hline', file=outPather, end='\n')

        s = 'h' + sep + np.array2string(hh, separator=sep, )
        s = s.replace('\n', '')
        s = s.replace('[', '')
        s = s.replace(']', '')
        print(s, file=outPather, end=end)
        print('\\\\\\hline', file=outPather)

        s = DofName + sep + np.array2string(Ndof, separator=sep, )
        s = s.replace('\n', '')
        s = s.replace('[', '')
        s = s.replace(']', '')
        print(s, file=outPather, end=end)
        print('\\\\\\hline', file=outPather)

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
            print(s, file=outPather, end=end)
            print('\\\\\\hline', file=outPather)

            if meshtype == 'tri':
                order = np.log(line[0:-1] / line[1:]) / np.log(2)
            else:
                # order = np.log(line[0:-1] / line[1:]) / np.log(hh[0:-1] / hh[1:])
                order = np.log(line[0:-1] / line[1:]) / np.log(Ndof[1:] / Ndof[0:-1])
            s = 'Order' + sep + '--' + sep + np.array2string(order, separator=sep, precision=2)
            s = s.replace('\n', '')
            s = s.replace('[', '')
            s = s.replace(']', '')
            print(s, file=outPather, end=end)
            print('\\\\\\hline', file=outPather)

        print('\\end{tabular}', file=outPather, end='\n')
        print('\\end{table}', file=outPather, end='\n')
        print('\n------------------------------------------------', file=outPather, end='\n')

        if flag:
            outPather.close()

    def showmultirate(self, k_slope, optionlist=None, lw=1, ms=4, propsize=10, outFlag=True):
        Ndof = self.Ndof
        errorMatrix = self.errorMatrix
        errorType = self.errorType
        out = self.out

        fig = plt.figure()
        fig.set_facecolor('white')
        axes = fig.gca()

        if optionlist is None:
            optionlist = ['k-*', 'r-o', 'b-D', 'g-->', 'k--8', 'm--x', 'r-.x', 'b-.+', 'b-.h', 'm:s', 'm:p', 'm:h']

        m, n = errorMatrix.shape
        k_slope = 0 if (k_slope < 0) else k_slope
        for i in range(m):
            if len(Ndof.shape) == 1:
                self.showrate(axes, k_slope, Ndof, errorMatrix[i], optionlist[i], label=errorType[i], lw=lw, ms=ms)
            else:
                self.showrate(axes, k_slope, Ndof[i], errorMatrix[i], optionlist[i], label=errorType[i], lw=lw, ms=ms)
        axes.legend(loc=3, framealpha=0.2, fancybox=True, prop={'size': propsize})

        plt.plot
        plt.savefig(out + '_rate.png') if (isinstance(out, str) & outFlag) else None
        # plt.show()
        plt.close()
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
