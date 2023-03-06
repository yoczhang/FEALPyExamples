#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: totest_plot_histogram.py
# @Author: Yongchao Zhang
# @Institution: Northwest University, Xi'an, Shaanxi, China
# @E-mail: yoczhang@126.com, yoczhang@nwu.edu.cn
# @Site: 
# @Time: Apr 07, 2022
# ---


import matplotlib  # 为了解决画图时采用 GUI (plt.show()) 的形式时, python3.8 崩溃的情况.
matplotlib.use("TkAgg")  # 为了解决画图时采用 GUI (plt.show()) 的形式时, python3.8 崩溃的情况.
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties


# font = FontProperties(fname="SimHei.ttf", size=14)
fig = plt.figure()
plt.title("Score distribution")
plt.ylabel("Number of people")  # 纵坐标label
section = ['<60', '60-69', '70-79', '80-89', '>90']
students = [2, 43, 32, 25, 4]
plt.bar(section, students)

length = len(students)
x1 = np.arange(length)

for a, b in zip(x1, students):
    plt.text(a, b + 0.1, '%.0f' % b, ha='center', va='bottom', fontsize=7)

plt.show()
