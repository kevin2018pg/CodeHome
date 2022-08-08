# -*- coding: utf-8 -*-
# @Time    : 2022/8/8 17:46
# @Author  : kevin
# @Version : python 3.7
# @Desc    : 最长上升子序列

class Solution:
    def lengthOfLIS(self, nums):
        if len(nums) <= 1:
            return len(nums)
        dp = [1] * len(nums)
        result = 0
        for i in range(1, len(nums)):
            for j in range(0, i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
            result = max(result, dp[i])  # 取长的子序列
        return result
