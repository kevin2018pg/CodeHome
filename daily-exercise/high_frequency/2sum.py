# -*- coding: utf-8 -*-
# @Time    : 2022/7/21 13:24
# @Author  : kevin
# @Version : python 3.7
# @Desc    : 2sum
# 哈希表存储每个数索引，查找target-当前数的差值
def two_sum(nums, target):
    hash_num_index = dict()
    for idx, nums in enumerate(nums):
        diff = target - nums
        if diff in hash_num_index:
            return [idx, hash_num_index[diff]]
        hash_num_index[nums] = idx
