#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: RestartLongTimeProgram.py
# @Author: Yongchao Zhang
# @Institution: Northwest University, Xi'an, Shaanxi, China
# @E-mail: yoczhang@126.com, yoczhang@nwu.edu.cn
# @Site: 
# @Time: Apr 30, 2022
# ---

__doc__ = """
目标: 保存尽量少的数据, 用来重新开始
"""

import pickle


class saveData:
    def __init__(self, filename):
        self.filename = filename

    def data_push(self, data):
        for k, v in data.items():
            self.__dict__[k] = v
        return None

    def data_save(self):
        f = open(self.filename, 'wb')
        f.write(pickle.dumps(ls))
        f.close()



class idol:
    def __init__(self, sing, dance, rap, basketball):
        self.sing = sing
        self.dance = dance
        self.rap = rap
        self.basketball = basketball


cxk = idol(10, 10, 10, 10)
ls = [cxk, cxk]
f = open('cxk_list.pkl', 'wb')
content = pickle.dumps(ls)
f.write(content)
f.close()


