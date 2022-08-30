# -*- coding: utf-8 -*-
# @Time    : 2022/7/12 16:43
# @Author  : kevin
# @Version : python 3.7
# @Desc    : longest-substring-without-repeating-characters

# 给定一个字符串 s ，请你找出其中不含有重复字符的 最长子串 的长度

class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        s_set = set()
        left, right, length = 0, 0, 0
        for i in range(len(s)):
            if i != 0:
                s_set.remove(s[left])
                left += 1
            while right < len(s) and s[right] not in s_set:
                s_set.add(s[right])
                right += 1
            length = max(length, right - left)

        return length


# 滑动窗口
def lengthOfLongestSubstring(s):
    str_map = set()
    right = -1
    length = 0
    for left in range(len(s)):
        if left != 0:
            str_map.remove(s[left - 1])
        while right < len(s) and s[right] not in str_map:
            str_map.add(s[right])
            right += 1
        length = max(length, right - left + 1)
    return length
