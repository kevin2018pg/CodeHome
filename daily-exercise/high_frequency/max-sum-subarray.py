# -*- coding: utf-8 -*-
# @Time    : 2022/7/19 21:30
# @Author  : kevin
# @Version : python 3.7
# @Desc    : 最大子数组和

# 给你一个整数数组 nums ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
# 子数组 是数组中的一个连续部分。
# 1、动态规划
# 2、贪心-状态传递：f(i)=max{f(i−1)+nums[i],nums[i]}

class Solution:
    def maxSubArray(self, nums):
        max_sum = nums[0]
        for i in range(1, len(nums)):
            if nums[i] + nums[i - 1] > nums[i]:  # 状态转移
                nums[i] += nums[i - 1]
            if nums[i] > max_sum:  # 更新max,全是负数的情况也会更新最大的一个负数
                max_sum = nums[i]
        return max_sum


class SolutionDp:
    def maxSubArray(self, nums):
        if not nums:
            return 0
        dp = [0] * len(nums)
        dp[0] = nums[0]
        result = nums[0]
        for i in range(1, len(nums)):
            dp[i] = max(dp[i - 1] + nums[i], nums[i])
            result = max(result, dp[i])
        return result
