# -*- coding: utf-8 -*-
# @Time : 2020/5/15 13:57
# @Author : Kevin
# @IDE: PyCharm
# @ModuleName : RotateMatrix
# @Description: 旋转 N*N 矩阵

def rotate_s1(matrix):
    """
    Do not return anything, modify matrix in-place instead.
    """
    length = len(matrix)
    ans = []
    for i in range(length):
        row = []
        for j in range(length):
            row.append(matrix[length - j - 1][i])
        ans.append(row)
    matrix[:] = ans
    return matrix


def rotate_s2(matrix):
    """
    Do not return anything, modify matrix in-place instead.
    """
    # matrix[::] = zip(*matrix[::-1])
    matrix[:] = list(map(list, zip(*matrix[::-1])))


def rotate_s3(matrix):
    """
    Do not return anything, modify matrix in-place instead.
    """
    N = len(matrix[0])
    for i in range(N):
        for j in range(i + 1, N):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    for i in range(N):
        matrix[i].reverse()


"""
四种方法:
一、遍历一维，对二维每个元素从原矩阵取，再合并，并赋给原矩阵
二、先上下翻转，然后zip将每一列取出来
三、同上，用map将zip转列表
四、对角线翻折，只需要交换非对角线上和非拐角点的值
"""
