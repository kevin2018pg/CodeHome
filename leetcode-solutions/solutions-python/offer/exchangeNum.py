# -*- coding: utf-8 -*-
# @Time : 2020/7/17 9:11
# @Author : Kevin
# @IDE: PyCharm
# @ModuleName : exchangeNum
# @Description: 交换数组数字，使得奇数在前半部分，偶数在后半部分


"""
输入一个整数数组，实现一个函数来调整该数组中数字的顺序，
使得所有奇数位于数组的前半部分，所有偶数位于数组的后半部分。

双指针
定义双指针 i ,j 分列数组左右两端
当i==j时退出
指针i遇到奇数执行i+=1，直到遇到偶数
指针j遇到偶数执行j-=1，直到遇到奇数
交换nums[i],nums[j]
"""

import collections


# x&1 位运算等价于 x%2取余运算，皆可用于判断数字奇偶性
class Solution:
    # 双指针1
    def exchange_s1(self, nums: List[int]) -> List[int]:
        i, j = 0, len(nums) - 1
        while i < j:
            while i < j and nums[i] % 2 == 1:
                i += 1
            while i < j and nums[j] % 2 == 0:
                j -= 1
            nums[i], nums[j] = nums[j], nums[i]
        return nums

    # 双指针2
    def exchange_s2(self, nums: List[int]) -> List[int]:
        i, j = 0, len(nums) - 1
        while i < j:
            if nums[i] % 2 == 0 and nums[j] % 2 == 1:
                nums[i], nums[j] = nums[j], nums[i]
            elif nums[i] % 2 == 1:
                i += 1
            elif nums[j] % 2 == 0:
                j -= 1
        return nums

    # 双端队列
    def exchange_s3(self, nums: List[int]) -> List[int]:
        tmp = collections.deque()
        for num in nums:
            tmp.appendleft(num) if num % 2 == 1 else tmp.append(num)
        return list(tmp)

    # 两个列表相加
    def exchange_s4(self, nums: List[int]) -> List[int]:
        num1 = []
        num2 = []
        for i in nums:
            if i % 2 == 1:
                num1.append(i)
            else:
                num2.append(i)
        return num1 + num2
