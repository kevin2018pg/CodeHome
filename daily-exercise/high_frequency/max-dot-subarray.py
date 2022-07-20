# -*- coding: utf-8 -*-
# @Time    : 2022/7/20 15:36
# @Author  : kevin
# @Version : python 3.7
# @Desc    : 最大子数组积

# 注意：不同于最大子数组和，乘法有正负性，正x负会变小，负x负会变大；
# 例如 a = {5, 6, -3, 4, -3}，对应的序列是{5, 30, -3, 4, -3}，
# 按最大子数组和贪心法，得到答案为 30，即前两个数的乘积，而实际上答案应该是全体数字的乘积

# tips：维护一个最大数和一个最小数，移动过程中分别与上一步最大和最小乘积，再与当前数求最大最小，更新结果
# fmax(i)=max{fmax(i−1)×ai, fmin(i−1)×ai, ai}
# fmin(i)=min{fmax(i−1)×ai, fmin(i−1)×ai, ai}

class Solution:
    def maxProduct(self, nums):
        max_num, min_num, res = nums[0], nums[0], nums[0]
        for i in range(1, len(nums)):
            tmp_max, tmp_min = max_num, min_num
            max_num = max(nums[i] * tmp_max, nums[i], nums[i] * tmp_min)
            min_num = min(nums[i] * tmp_min, nums[i], nums[i] * tmp_max)
            res = max(res, max_num)
        return res
