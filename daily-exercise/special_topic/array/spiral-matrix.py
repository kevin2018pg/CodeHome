# -*- coding: utf-8 -*-
# @Time    : 2022/8/12 9:34
# @Author  : kevin
# @Version : python 3.7
# @Desc    : spiral-matrix

# 螺旋矩阵1
class Solution:
    def spiralOrder(self, matrix):
        if len(matrix) == 0 or len(matrix[0]) == 0:
            return []

        # 行、列
        row, column = len(matrix), len(matrix[0])
        # 初始化数组
        res = [0] * row * column
        # 起始点
        startx, starty = 0, 0
        # 循环层数，和中间点
        loop, mid = min(row, column) // 2, min(row, column) // 2
        # 计数
        count = 0

        for offset in range(1, loop + 1):  # 每循环一层偏移量加1，偏移量从1开始
            for i in range(starty, column - offset):  # 从左至右，左闭右开
                res[count] = matrix[startx][i]
                count += 1
            for i in range(startx, row - offset):  # 从上至下
                res[count] = matrix[i][column - offset]
                count += 1
            for i in range(column - offset, starty, -1):  # 从右至左
                res[count] = matrix[row - offset][i]
                count += 1
            for i in range(row - offset, startx, -1):  # 从下至上
                res[count] = matrix[i][starty]
                count += 1
            # 更新起始点
            startx += 1
            starty += 1
        # 出现行列为奇数时，需要填充中心行或者中心列
        if min(row, column) % 2 != 0:
            if row > column:  # 行大于列，填充中心行
                for i in range(mid, mid + row - column + 1):
                    res[count] = matrix[i][mid]
                    count += 1
            else:  # 行小于列，填充中心列
                for i in range(mid, mid + column - row + 1):
                    res[count] = matrix[mid][i]
                    count += 1
        return res


test = Solution()
res = test.spiralOrder([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(res)
