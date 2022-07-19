# -*- coding: utf-8 -*-
# @Time : 2020/5/7 15:53
# @Author : Kevin
# @IDE: PyCharm
# @ModuleName : 01.TwoSum
# @Description: 两数之和
"""
给定一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值的那 两个 整数，并返回他们的数组下标。
你可以假设每种输入只会对应一个答案。但是，数组中同一个元素不能使用两遍。

给定 nums = [2, 7, 11, 15], target = 9
因为 nums[0] + nums[1] = 2 + 7 = 9
所以返回 [0, 1]
"""


def TwoSum_s1(nums, target):
    for i in range(len(nums) - 1):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return i, j


def TwoSum_s2(nums, target):
    for i in range(len(nums)):
        other = target - nums[i]
        if other in nums[i + 1:]:
            return i, nums[i + 1:].index(other) + (i + 1)


def TwoSum_s3(nums, target):
    hashmap = {}
    for ind, num in enumerate(nums):
        hashmap[num] = ind
    for i, num in enumerate(nums):
        j = hashmap.get(target - num)
        if j is not None and i != j:
            return i, j


def TwoSum_s4(nums, target):
    hashmap = {}
    for ind, num in enumerate(nums):
        diff = target - num
        if diff in hashmap:
            return ind, hashmap.get(diff)
        hashmap[num] = ind

# a, b = TwoSum_s1([2, 7, 11, 15], 26)
# a1, b1 = TwoSum_s2([2, 7, 11, 15], 26)
# a1, b1 = TwoSum_s3([2, 7, 11, 15], 26)
# a1, b1 = TwoSum_s4([2, 7, 11, 15], 26)
# 最后两种哈希方法无差别，时间复杂度极低，但是内存消耗稍大，最慢的是第一种方法
