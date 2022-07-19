# -*- coding: utf-8 -*-
# @Time : 2020/6/6 23:20
# @Author : Kevin
# @IDE: PyCharm
# @ModuleName : 0606LongestConsecutiveSequence
# @Description: 最长连续序列

"""
给定一个未排序的整数数组，找出最长连续序列的长度。
要求算法的时间复杂度为 O(n)。

输入: [100, 4, 200, 1, 3, 2]
输出: 4
解释: 最长连续序列是 [1, 2, 3, 4]。它的长度为 4。
"""


# 先对列表去重
# 遍历，判断比当前数小1的在不在数组，在跳出当前循环
# 判循环判断当前数累加1是否在数组里，并计算长度
# 不断更新最长连续序列长度

class Solution1:
    def longestConsecutive(self, nums: List[int]) -> int:
        nums = set(nums)
        ans = 0
        for n in nums:
            if n - 1 in nums:
                continue
            curlen = 1
            while n + 1 in nums:
                curlen += 1
                n += 1
            ans = max(ans, curlen)
        return ans


class Solution2:
    def longestConsecutive(self, nums: List[int]) -> int:
        nums = set(nums)
        res = 0
        for num in nums:
            # 判断是否是第一个数字
            if num - 1 not in nums:
                tmp = 1
                while num + 1 in nums:
                    num += 1
                    tmp += 1
                res = max(res, tmp)
        return res
