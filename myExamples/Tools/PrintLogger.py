#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: PrintLogger.py
# @Author: Yongchao Zhang, Northwest University, Xi'an
# @E-mail: yoczhang@nwu.edu.cn
# @Site:
# @Time: Jul 12, 2021
# ---

import sys
import os
import sys
import io
import datetime


def create_detail_day():
    '''

    :return:
    '''
    # 年-月-日
    # daytime = datetime.datetime.now().strftime('day'+'%Y-%m-%d')
    # 年_月_日
    daytime = datetime.datetime.now().strftime('%Y%m%d')
    # 时：分：秒
    # hourtime = datetime.datetime.now().strftime("%H:%M:%S")
    hourtime = datetime.datetime.now().strftime("%H%M%S")
    # detail_time = daytime
    # print(daytime + "-" + hourtime)
    detail_time = daytime + "-" + hourtime
    return detail_time


def make_print_to_file(filename="Default.log", path='./'):
    '''
     example:
    use  make_print_to_file() ,  and the   all the information of funtion print , will be write in to a log file
    :param path:  the path to save print information
    :return:
    '''

    class Logger:
        def __init__(self, filename="Default.log", path="./"):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "a", encoding='utf8')

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

    sys.stdout = Logger(filename=filename + '-' + create_detail_day() + '.log', path=path)

    # print(create_detail_time().center(60, '*'))


# # --- to test --- # #
if __name__ == '__main__':
    make_print_to_file(filename='lala', path="/Users/yczhang/Documents/FEALPy/FEALPyExamples/FEALPyExamples/myExamples/Logs/")

    print('explanation'.center(80, '*'))
    info1 = '从大到小排序'
    info2 = ' sort the form large to small'
    print(info1)
    print(info2)
    print('END:  explanation'.center(80, '*'))
