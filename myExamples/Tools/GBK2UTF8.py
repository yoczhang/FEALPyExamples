#!/anaconda3/envs/FEALPy/bin python3.8
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: GBK2UTF8.py
# @Author: Yongchao Zhang
# @Institution: Northwest University, Xi'an, Shaanxi, China
# @E-mail: yoczhang@126.com, yoczhang@nwu.edu.cn
# @Site: 
# @Time: May 07, 2022
# ---


import fnmatch
import os
import sys
import codecs
import chardet


def FindFile(path, fnexp="*.m"):
    for root, dirs, files in os.walk(path):
        for filename in fnmatch.filter(files, fnexp):
            yield os.path.join(root, filename)


def ReadFile(filePath, encoding="gbk"):
    with codecs.open(filePath, "r", encoding) as f:
        return f.read()


def WriteFile(filePath, u, encoding="utf-8"):
    with codecs.open(filePath, "w", encoding) as f:
        f.write(u)


def GBK_2_UTF8(src, dst):
    content = ReadFile(src, encoding="gbk")
    WriteFile(dst, content, encoding="utf-8")


def UTF8_2_GBK(src, dst):
    content = ReadFile(src, encoding="utf-8")
    WriteFile(dst, content, encoding="gbk")


def trans(f_path):
    print(f_path)
    for fn in FindFile(f_path):
        print(fn)
        try:
            responseStr = open(fn, "r").read()
        except UnicodeDecodeError:
            responseStr = open(fn, "rb").read()

        try:
            responseStr = responseStr.encode()
        except AttributeError:
            pass

        d = chardet.detect(responseStr)
        print(d)
        if d['encoding'] != 'utf-8':
            GBK_2_UTF8(fn, fn)
            print("ok")


if __name__ == '__main__':
    # trans("./")

    # |--------------------- test ---------------------| #
    # if len(sys.argv) > 1 :
    #     print("len(sys.argv) = ", len(sys.argv))
    #     print(sys.argv[:])
    #     for fpath in sys.argv[1:]:
    #         print("fpath=", fpath)
    #         print("type(fpath): ", type(fpath))
    # |--------------------- test ---------------------| #

    if len(sys.argv) > 1:
        for fpath in sys.argv[1:]:
            trans(fpath)
    else:
        print("You should input the path of the files")
        print("And note that the input path cannot be str, such as './', just input ./ ")
        fpath = input("path: ")
        trans(fpath)
