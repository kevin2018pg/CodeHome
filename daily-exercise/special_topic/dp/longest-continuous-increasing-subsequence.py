# -*- coding: utf-8 -*-
# @Time    : 2022/8/8 20:33
# @Author  : kevin
# @Version : python 3.7
# @Desc    : 最长连续递增子序列

# 概括来说：不连续递增子序列的跟前0-i 个状态有关，连续递增的子序列只跟前一个状态有关

class Solution:
    def findLengthOfLCIS(self, nums):
        if len(nums) <= 1:
            return len(nums)
        dp = [1] * len(nums)
        res = 1
        for i in range(1, len(nums)):
            if nums[i] > nums[i - 1]:
                dp[i] = dp[i - 1] + 1
            res = max(res, dp[i])
        return res
