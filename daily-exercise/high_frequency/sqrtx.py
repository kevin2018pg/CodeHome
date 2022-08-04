# -*- coding: utf-8 -*-
# @Time    : 2022/8/4 21:41
# @Author  : kevin
# @Version : python 3.7
# @Desc    : x的平方根（不允许使用内置函数）


class Solution:
    def mySqrt(self, x):  # 二分查找法
        l, r, ans = 0, x, 0
        while l <= r:
            mid = l + (r - l) // 2
            if mid * mid <= x:  # 平方根整数满足此条件
                ans = mid
                l = mid + 1
            else:
                r = mid - 1
        return ans
