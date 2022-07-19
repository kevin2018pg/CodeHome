# -*- coding: utf-8 -*-
# @Time : 2020/6/2 20:16
# @Author : Kevin
# @IDE: PyCharm
# @ModuleName : lackofnumber
# @Description: 缺失的数字


"""
数学求和法求解，两和之差就是缺失的数字
排序数组的搜索问题转换为二分查找
"""


def missingNumber_s1(nums):
    sum01 = len(nums) * (1 + len(nums)) / 2
    sum02 = sum(nums)
    return sum01 - sum02


"""
定义 left:=0 right:=len(nums) 用来确定数组的 mid
因为nums是有序数组，如果mid下标的值和mid不相同就在左边查找
如果 nums[mid]==mid ，说明左边是连续的有序数组，缺失的数字就在右边查找
"""


def missingNumber_s2(nums):
    i, j = 0, len(nums) - 1
    while i <= j:
        m = (i + j) // 2
        if nums[m] == m:
            i = m + 1
        else:
            j = m - 1
    return i
