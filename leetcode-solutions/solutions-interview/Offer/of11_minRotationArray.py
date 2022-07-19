'''
Author: your name
Date: 2020-11-18 17:43:48
LastEditTime: 2020-11-19 14:15:36
LastEditors: Please set LastEditors
Description: 剑指 Offer 11. 旋转数组的最小数字

把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。输入一个递增排序的数组的一个旋转，输出旋转数组的最小元素。例如，数组 [3,4,5,1,2] 为 [1,2,3,4,5] 的一个旋转，该数组的最小值为1。  

示例 1：

输入：[3,4,5,1,2]
输出：1


FilePath: \learningflow\Review\OfferSolution\of11_minArray.py
'''


# 寻找数组最小旋转点
class Solution:
    # 二分查找方法
    def minArray_01(self, numbers: List[int]) -> int:
        i, j = 0, len(numbers) - 1  # 先确定上下界
        # 当上下界相同，二分查找结束
        while i < j:
            # 二分点pivot
            m = (i + j) // 2  # i 和 j 都很大的时候可能会int溢出，比较好的写法是 i + (j - i) // 2
            # pivot大于上界，旋转点在m右侧
            if numbers[m] > numbers[j]:
                i = m + 1
            # pivot小于上确界，旋转点在m左侧，包括自身（pivot可能是旋转点）
            elif numbers[m] < numbers[j]:
                j = m
            # 相等无法判断，上确界缩小一位比较
            else:  # 重复元素的存在
                j -= 1

        return numbers[i]

    # 遍历方法，暴力
    def minArray_02(self, numbers: List[int]) -> int:
        for i in range(len(numbers) - 1):
            if numbers[i] > numbers[i + 1]:
                return numbers[i + 1]
        return numbers[0]
