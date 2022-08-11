# -*- coding: utf-8 -*-
# @Time    : 2022/8/11 13:53
# @Author  : kevin
# @Version : python 3.7
# @Desc    : squares-of-a-sorted-array

# 升序数组平方后再升序排列，需要返回新的数组
# 常规方法：先平方，再排序
# 双指针法：首尾指针
class Solution:
    def sortedSquares(self, nums):
        n = len(nums)
        head, tail = 0, n - 1
        k = n - 1
        res = [-1] * n
        while head <= tail:
            # 先计算首尾平方
            h2 = nums[head] ** 2
            t2 = nums[tail] ** 2
            if h2 < t2:
                res[k] = t2
                tail -= 1
            else:
                res[k] = h2
                head += 1
            k -= 1
        return res
