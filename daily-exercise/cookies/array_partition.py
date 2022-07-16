# -*- coding: utf-8 -*-
# @Time    : 2022/3/10 14:18
# @Author  : kevin
# @File    : array_partition.py
# @Version : python 3.6
# @Desc    : 数组划分


# 给出一个整数数组 nums 和一个整数 k。划分数组（即移动数组 nums 中的元素），使得所有小于k的元素移到左边，所有大于等于k的元素移到右边，返回数组划分的位置，即数组中第一个位置 i，满足 nums[i] 大于等于 k。
def big_small_partition(nums, k):
    # write your code here
    if (nums == None or len(nums) == 0): return 0
    start, end = 0, len(nums) - 1
    while start < end:
        while start < end and nums[start] < k:
            start += 1
        while start < end and nums[end] >= k:
            end -= 1
        if nums[start] >= k and nums[end] < k:
            nums[start], nums[end] = nums[end], nums[start]
            start += 1
            end -= 1
    print(nums)
    if nums[start] >= k:
        return start
    return start + 1


def big_small_partition_1(nums, k):
    # write your code here
    if (nums == None or len(nums) == 0): return 0
    start, end = 0, len(nums) - 1
    while start < end:
        if start < end and nums[start] < k:
            start += 1
        elif start < end and nums[end] >= k:
            end -= 1
        elif nums[start] >= k and nums[end] < k:
            nums[start], nums[end] = nums[end], nums[start]
            start += 1
            end -= 1
    print(nums)
    if nums[start] >= k:
        return start
    return start + 1


def ji_ou_partition(nums):
    if (nums == None or len(nums) == 0): return 0
    start, end = 0, len(nums) - 1
    while start < end:
        while (start < end and nums[start] % 2 == 1):
            start += 1
        while (start < end and nums[end] % 2 == 0):
            end -= 1
        if (nums[start] % 2 == 0 and nums[end] % 2 == 1):
            nums[start], nums[end] = nums[end], nums[start]
            start += 1
            end -= 1
    return nums


# pos = big_small_partition([1, 4, 2, 5, 4, 8, 1, 0, 5], 5)
pos1 = big_small_partition_1([3, 4, 7, 9, 6, 2, 8, 5], 5)
# nums = ji_ou_partition([1, 4, 2, 5, 4, 8, 1, 0, 5])
# print(pos)
# print(pos1)
# print(nums)
