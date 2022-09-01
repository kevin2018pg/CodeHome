# -*- coding: utf-8 -*-
# @Time    : 2022/9/1 21:23
# @Author  : kevin
# @Version : python 3.7
# @Desc    : 反转字符串


class Solution:
    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        left, right = 0, len(s) - 1
        while left < right:
            s[left], s[right] = s[right], s[left]
            left += 1
            right -= 1

    def reverseString2(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        length = len(s)
        mid = length // 2 - 1
        for i in range(mid, -1, -1):
            j = length - i - 1
            s[i], s[j] = s[j], s[i]
