# -*- coding: utf-8 -*-
# @Time : 2020/6/5 12:49
# @Author : Kevin
# @IDE: PyCharm
# @ModuleName : 0605spiralOrderMatrix
# @Description: 从外向里以顺时针的顺序依次打印出每一个数字。

"""
输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
输出：[1,2,3,6,9,8,7,4,5]
"""


# 本题采用模拟法去做
def spiralOrder(matrix):
    if not matrix: return []
    l, r, t, b, res = 0, len(matrix[0]) - 1, 0, len(matrix) - 1, []
    while True:
        for i in range(l, r + 1): res.append(matrix[t][i])  # left to right
        t += 1
        if t > b: break
        for i in range(t, b + 1): res.append(matrix[i][r])  # top to bottom
        r -= 1
        if l > r: break
        for i in range(r, l - 1, -1): res.append(matrix[b][i])  # right to left
        b -= 1
        if t > b: break
        for i in range(b, t - 1, -1): res.append(matrix[i][l])  # bottom to top
        l += 1
        if l > r: break
    return res


"""
解题思路：模拟法
1、空值处理： 当 matrix 为空时，直接返回空列表 [] 即可。
2、初始化： 矩阵 左、右、上、下 四个边界 l , r , t , b ，用于打印的结果列表 res 。
3、循环打印： “从左向右、从上向下、从右向左、从下向上” 四个方向循环，每个方向打印中做以下三件事；
    1）根据边界打印，即将元素按顺序添加至列表 res 尾部；
    2）边界向内收缩 1（代表已被打印）；
    3）判断是否打印完毕（边界是否相遇），若打印完毕则跳出。
4、返回值： 返回 res 即可。
"""
