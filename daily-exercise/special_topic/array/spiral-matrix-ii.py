# -*- coding: utf-8 -*-
# @Time    : 2022/8/11 21:24
# @Author  : kevin
# @Version : python 3.7
# @Desc    : spiral-matrix-ii

class Solution:
    def generateMatrix(self, n):
        nums = [[0] * n for _ in range(n)]
        # 迭代次数
        loop, mid = n // 2, n // 2  # 迭代次数为n整除2，奇数时有一个中心点，坐标也是n整除2
        # 起始点
        startx, starty = 0, 0
        # 计数
        count = 1
        for offset in range(1, loop + 1):  # 每循环一层偏移量加1，偏移量从1开始
            for i in range(starty, n - offset):  # 从左至右，左闭右开
                nums[startx][i] = count
                count += 1
            for i in range(startx, n - offset):  # 从上至下
                nums[i][n - offset] = count
                count += 1
            for i in range(n - offset, starty, -1):  # 从右至左
                nums[n - offset][i] = count
                count += 1
            for i in range(n - offset, startx, -1):  # 从下至上
                nums[i][starty] = count
                count += 1
            startx += 1  # 更新起始点
            starty += 1
        if n % 2 != 0:  # n为奇数时，填充中心点
            nums[mid][mid] = count
        return nums
