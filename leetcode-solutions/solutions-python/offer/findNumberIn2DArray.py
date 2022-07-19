# -*- coding: utf-8 -*-
# @Time : 2020/6/12 21:05
# @Author : Kevin
# @IDE: PyCharm
# @ModuleName : findNumberIn2DArray
# @Description:  二维数组中的查找（暴力法+标志位法）
class Solution:
    def findNumberIn2DArray_s1(self, matrix: List[List[int]], target: int) -> bool:
        for row in matrix:
            for column in row:
                if column == target:
                    return True
        return False

    def findNumberIn2DArray_s2(self, matrix: List[List[int]], target: int) -> bool:
        # 可以选取的标志位有左下角和右上角
        row, col = len(matrix) - 1, 0
        while row >= 0 and col < len(matrix[0]):
            if matrix[row][col] > target:
                row -= 1
            elif matrix[row][col] < target:
                col += 1
            # elif matrix[row][col] == target:
            #     return True
            else:
                return True
        return False
